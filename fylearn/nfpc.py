# -*- coding: utf-8 -*-
"""
Fuzzy pattern classifier with negative and positive examples


References:
-----------
[1] Davidsen 2015.

"""

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.neighbors import DistanceMetric
import fuzzylogic as fl

logger = logging.getLogger("fylearn.nfpc")

def pi_factory(**kwargs):
    m = kwargs["m"] if "m" in kwargs else 2.0
    c = kwargs["mean"]
    d = (kwargs["max"] - kwargs["min"]) / 2.0
    return fl.PiSet(a=c - d, r=c, b=c + d, m=m)

def t_factory(**kwargs):
    c = kwargs["mean"]
    d = (kwargs["max"] - kwargs["min"]) / 2.0
    return fl.TriangularSet(c - d, c, c + d)

def distancemetric_f(name, **kwargs):
    """
    Factory for a distance metric supported by DistanceMetric
    """
    def _distancemetric_factory(X):
        return DistanceMetric.get_metric(name)
    return _distancemetric_factory

#
# Authors: Søren Atmakuri Davidsen <sorend@gmail.com>
#

def discretize(X, n_bins=5):
    """
    Parameters:
    -----------
    n_bins : number of bins per dimension to discretize into.

    X : training dataset.

    y : training classes.
    """
    d_min, d_max = np.min(X, 0), np.max(X, 0)

    # M = np.zeros(X.shape)

    for i in range(X.shape[1]):  # for each dimension
        ls = np.linspace(d_min[i], d_max[i], n_bins)
        diff = ls[1] - ls[0]
        s_j = [ fl.TriangularSet(x - diff, x, x + diff) for x in ls ]

        R = np.zeros((X.shape[0], n_bins))
        for j, s in enumerate(s_j):
            R[:, j] = s(X[:, i])

        M = np.argmax(R, 1)  # take the set with the maximum support
        print "R", R
        print "M", M

    pass

def shrink_for_feature(C, factory, alpha_cut, iterations):
    """
    Performs shrinking for a single dimension (feature)
    """

    def create_mu(idx):
        s = sum(idx) > 0
        tmin = np.nanmin(C[idx]) if s else 0
        tmax = np.nanmax(C[idx]) if s else 1
        tmean = np.nanmean(C[idx]) if s else 0.5
        return factory(min=tmin, mean=tmean, max=tmax)

    C_idx = C >= 0  # create mask
    mu = create_mu(C_idx)
    # mu_orig = mu

    for iteration in range(iterations):
        A = mu(C)
        C_idx = A >= alpha_cut
        mu = create_mu(C_idx)

    # print "mu_orig", mu_orig, "mu_final", mu

    return mu

def build_shrinking_memberships(X, factory, alpha_cut, iterations):
    # shrink each dimension (feature)
    return [ shrink_for_feature(X[:, i], factory, alpha_cut, iterations) for i in range(X.shape[1]) ]

def predict_mus(X, mus, aggregation):
    A = np.zeros(X.shape)
    for c_idx in range(X.shape[1]):
        A[:, c_idx] = mus[c_idx](X[:, c_idx])

    y = np.zeros(len(X))
    for r_idx in range(X.shape[0]):
        y[r_idx] = aggregation(A[r_idx, :])

    return y

def evaluate_prototype(x, X, X_missing, mus):
    """
    Evaluates fitness of prototype.
    """
    return 1.0

def build_extras(X, factory, mus, aggregation, distance):
    y_proba = predict_mus(X, mus, aggregation)
    X_missing = X[y_proba <= 0.0, :]
    print "X_missing", X_missing

    X_w = np.array([ evaluate_prototype(x, X, X_missing) for x in X_missing ])

    
    # return [ extra_for_feature(X[:, i], factory, mus[i]) for i in range(X.shape[1]) ]

class ShrinkingPositiveFuzzyPatternClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, aggregation=np.mean,
                 membership_factory=pi_factory,
                 alpha_cut=0.25,
                 iterations=3,
                 **kwargs):
        """
        Constructs classifier

        Params:
        -------
        positive_class : the class to use as positive. None means take first class (lowest value).

        aggregation : Aggregation to use in the classifier

        membership_factory : factory to construct membership functions.

        alpha_cut : alpha cut to make of the learned set.

        iterations : number of times to take alpha cut.

        """
        self.aggregation = aggregation
        self.membership_factory = membership_factory
        self.alpha_cut = float(alpha_cut)
        self.iterations = int(iterations)

    def get_params(self, deep=False):
        return {"aggregation": self.aggregation,
                "membership_factory": self.membership_factory,
                "alpha_cut": self.alpha_cut,
                "iterations": self.iterations}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y=None):
        X = check_array(X)

        # if y is not None:
        #    raise ValueError("y should not be given for one-class-classifier")

        # build membership functions for each feature for each class
        self.mus_ = build_shrinking_memberships(X,
                                                self.membership_factory,
                                                self.alpha_cut,
                                                self.iterations)

        # build extras for each class
        self.extras_ = build_extras(X,
                                    self.membership_factory,
                                    self.mus_,
                                    self.aggregation)

        logger.info("mus_ %s" % (repr(self.mus_),))
        logger.info("extras_ %s" % (repr(self.extras_),))

        return self

    def predict(self, X):
        """
        Predicts if examples in X belong to classifier's class or not.

        Parameters
        ----------
        X : examples to predict for.
        """
        y_proba = self.predict_proba(X)
        y = np.zeros(len(y_proba))
        y[y_proba > 0.0] = 1
        y[y_proba == 0.0] = -1
        return y

    def predict_proba(self, X):
        if not hasattr(self, "mus_"):
            raise Exception("Perform a fit first.")

        X = check_array(X)

        if X.shape[1] != len(self.mus_):
            raise ValueError("Number of features do not match trained number of features")

        return predict_mus(X, self.mus_, self.aggregation)
