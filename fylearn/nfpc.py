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
from sklearn.preprocessing import normalize
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

def predict_protos(X, protos, aggregation):
    y = np.zeros((X.shape[0], len(protos)))
    A = np.zeros(X.shape)
    for clz_no, proto in enumerate(protos):
        for col_no in range(X.shape[1]):
            A[:, col_no] = proto[col_no](X[:, col_no])
        # print "A", A
        # print "y", y[:, clz_no]
        y[:, clz_no] = aggregation(A, axis=1)
    return y


class IterativeShrinking:
    def __init__(self, iterations=3, alpha_cut=0.1):
        self.iterations = iterations
        self.alpha_cut = alpha_cut

    def __call__(self, X, mu_factory):
        return [ self.shrink_for_feature(X[:, i], mu_factory, self.alpha_cut, self.iterations)
                 for i in range(X.shape[1]) ]

    def shrink_for_feature(self, C, factory, alpha_cut, iterations):
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

        return mu


class ShrinkingFuzzyPatternClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, aggregation=np.mean,
                 membership_factory=pi_factory,
                 shrinking=IterativeShrinking(3, 0.85),
                 arg_select=np.nanargmax,
                 **kwargs):
        """
        Constructs classifier

        Params:
        -------
        aggregation : Aggregation to use in the classifier.

        membership_factory : factory to construct membership functions.

        shrinking : function to use for shrinking.

        arg_select : function to select class from aggregated mu values.

        """
        self.aggregation = aggregation
        self.membership_factory = membership_factory
        self.shrinking = shrinking
        self.arg_select = arg_select

    def get_params(self, deep=False):
        return {"aggregation": self.aggregation,
                "membership_factory": self.membership_factory,
                "shrinking": self.shrinking,
                "arg_select": self.arg_select}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        X = check_array(X)

        self.classes_, y = np.unique(y, return_inverse=True)

        if np.nan in self.classes_:
            raise ValueError("nan not supported for class values")

        # build membership functions for each feature for each class
        self.protos_ = [
            self.shrinking(X[y == idx], self.membership_factory)
            for idx, class_value in enumerate(self.classes_)
        ]

        return self

    def predict(self, X):
        """
        Predicts if examples in X belong to classifier's class or not.

        Parameters
        ----------
        X : examples to predict for.
        """
        if not hasattr(self, "protos_"):
            raise Exception("Perform a fit first.")

        y_mu = predict_protos(X, self.protos_, self.aggregation)

        print "X", X.shape
        print "predicted", y_mu
        print "take", self.arg_select(y_mu, 1)

        return self.classes_.take(self.arg_select(y_mu, 1))

    def predict_proba(self, X):
        if not hasattr(self, "protos_"):
            raise Exception("Perform a fit first.")

        X = check_array(X)

        if X.shape[1] != len(self.mus_):
            raise ValueError("Number of features do not match trained number of features")

        y_mu = predict_protos(X, self.protos_, self.aggregation)

        return 1.0 - normalize(y_mu, 'l1')
