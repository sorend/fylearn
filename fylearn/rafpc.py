# -*- coding: utf-8 -*-
"""Random agreement Fuzzy Pattern Classifier method.

The module structure is the following:

- The "RandomAgreementFuzzyPatternClassifier" implements the model learning using the [1] algorithm.

References:

[1] Davidsen, 2014.
"""
import logging
import numpy as np
import scipy.stats as stats

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

import fylearn.fuzzylogic as fl

def agreement_t_test(a, b):
    """ Check agreement based on means of two samples, using the t-statistic. """
    means1 = np.nanmean(a, 0)

    t1, p1 = stats.ttest_1samp(b, means1)
    # t2, p2 = stats.ttest_1samp(a, means2)

    # select agreeing featurs (p <= 0.05)
    return p1 < 0.05

def fuzzify_partitions(p):
    def fuzzify_p(A):
        R = np.zeros((A.shape[0], A.shape[1] * p))

        cmin, cmax = np.nanmin(A, 0), np.nanmax(A, 0)
        psize = (cmax - cmin) / (p - 1)

        mus = []
        # iterate features
        for i in range(A.shape[1]):
            # iterate partitions
            mu_i = []
            offset = cmin[i]
            for j in range(p):
                f = fl.TriangularSet(offset - psize[i], offset, offset + psize[i])
                R[:, (i * p) + j] = f(A[:, i])
                mu_i.append(f)
                offset += psize[i]
            mus.append(mu_i)
        return p, R, mus
    return fuzzify_p

def fuzzify_mean(A):
    # output for fuzzified values
    R = np.zeros((A.shape[0], A.shape[1] * 3))

    cmin, cmax, cmean = np.nanmin(A, 0), np.nanmax(A, 0), np.nanmean(A, 0)
        
    left = np.array([cmin - (cmax - cmin), cmin, cmax]).T
    middle = np.array([cmin, cmean, cmax]).T
    right = np.array([cmin, cmax, cmax + (cmax - cmin)]).T

    mus = []

    for i in range(A.shape[1]):
        f_l = fl.TriangularSet(*left[i])
        f_m = fl.TriangularSet(*middle[i])
        f_r = fl.TriangularSet(*right[i])
        R[:,(i*3)] = f_l(A[:,i])
        R[:,(i*3)+1] = f_m(A[:,i])
        R[:,(i*3)+2] = f_r(A[:,i])
        mus.extend([(i, f_l), (i, f_m), (i, f_r)])

    return 3, R, mus

def agreement_fuzzy(aggregation, A, B):
    """
    Calculate agreement between two samples.

    A  :  First sample
    B  :  Second sample
    """

    # avg values of samples (column wise)
    S_A, S_B = np.nanmean(A, 0), np.nanmean(B, 0)

    d = 1.0 - ((S_A - S_B) ** 2)
    a = aggregation(d)

    return a, d

def agreement_hamming(p, X, a, b):
    d = np.abs(X[a, :] - X[b, :])
    f = int(X.shape[1] / p)
    E = np.zeros(f)
    for i in range(f):
        E[i] = np.sum(d[(i * p):(i * p) + p])
    return 1.0 - ((1.0 / p) * E)

def triangular_factory(*args):
    return fl.TriangularSet(args[0], args[1], args[2])

def pi_factory(*args):
    return fl.PiSet(a=args[0], r=args[1], b=args[2], m=2.0)

def build_memberships(X, factory):
    mins = np.nanmin(X, 0)
    maxs = np.nanmax(X, 0)
    means = np.nanmean(X, 0)
    return [ (i, factory(means[i] - ((maxs[i] - mins[i]) / 2.0),
                         means[i], means[i] + ((maxs[i] - mins[i]) / 2.0))) for i in range(X.shape[1]) ]

def agreement_pruning(X, proto, n_features, rs):

    if len(proto) <= n_features:  # nothing to prune.
        return proto

    # prune from random samples
    for S in X[rs.choice(len(X), len(proto) - n_features)]:
        y = np.array([p(S[idx]) for idx, p in proto ])  # evaluate sample using the prototype
        worst = np.argsort(y)                           # find worst
        del proto[worst[0]]                             # filter worst

    # print "proto-after", proto
    return proto

def build_for_class(X, max_samples, n_features, rs, factory):

    # construct wanted number of prototypes
    max_no = max(max_samples, len(X))
    sample_idx = rs.permutation(max_no) % len(X)

    # construct memberships for all features based on the sample
    proto = build_memberships(X, factory)

    return agreement_pruning(X, proto, n_features, rs)

def build_for_class_multi(X, max_samples, n_features, rs, factory, n_protos):

    protos = []
    for p in range(n_protos):
        # construct wanted number of prototypes
        max_no = max(max_samples, len(X))
        sample_idx = rs.permutation(max_no) % len(X)

        # construct memberships for all features based on the sample
        proto = build_memberships(X[sample_idx], factory)
        # perform pruning
        proto = agreement_pruning(X, proto, n_features, rs)
        # add to list of protos for the class
        protos.append(proto)

    return protos

def _predict(prototypes, aggregation, classes, X, n_features):
    Mus = np.zeros((X.shape[0], n_features))
    R = np.zeros((X.shape[0], len(classes))) # holds output for each class
    attribute_idxs = range(n_features)

    # class_idx has class_prototypes membership functions
    for class_idx, class_prototypes in prototypes.items():
        for i in attribute_idxs:
            fidx, cp = class_prototypes[i]
            Mus[:, i] = cp(X[:, fidx])
        R[:, class_idx] = aggregation(Mus)

    return classes.take(np.argmax(R, 1))

def _predict_multi(prototypes, aggregation, classes, X, n_features):

    Mus = np.zeros(X.shape)                   # holds output per prototype
    R = np.zeros((X.shape[0], len(classes)))  # holds output for each class
    feature_nos = range(n_features)           # index for agreements

    # class_idx has class_prototypes membership functions
    for class_idx, class_prototypes in prototypes.items():
        C = np.zeros((X.shape[0], len(class_prototypes)))
        for j, cp in enumerate(class_prototypes):
            for i in feature_nos:
                f_idx, mu_f = cp[i]
                Mus[:, i] = mu_f(X[:, f_idx])
            C[:, j] = aggregation(Mus)
        R[:, class_idx] = np.max(C, 1)

    return classes.take(np.argmax(R, 1))


logger = logging.getLogger("rafpc")

class RandomAgreementFuzzyPatternClassifier(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {"n_protos": self.n_protos,
                "n_features": self.n_features,
                "max_samples": self.max_samples,
                "epsilon": self.epsilon,
                "aggregation": self.aggregation,
                "membership_factory": self.membership_factory,
                "random_state": self.random_state}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.setattr(key, value)
        return self

    def __init__(self, n_protos=5, n_features=None,
                 max_samples=100, epsilon=0.95,
                 aggregation=fl.mean, membership_factory=triangular_factory,
                 random_state=None):
        """
        Initialize the classifier

        Parameters:
        -----------
        n_protos : the number of prototypes to keep for each class.

        n_features : the number of features to include in each prototype.
                     None means use all features.

        max_samples : the number of samples to draw in finding agreement.

        epsilon : the minimum agreement needed before eliminiation.

        aggregation : The aggregation to use for inference.

        membership_factory : the factory to create membership functions.

        random_state : The random state to use for drawing samples.
                       None means no specific random state.
        
        """
        self.n_protos = n_protos
        self.n_features = n_features
        self.max_samples = max_samples
        self.epsilon = epsilon
        self.aggregation = aggregation
        self.membership_factory = membership_factory
        self.random_state = random_state

    def fit(self, X, y):

        # get random
        rs = check_random_state(self.random_state)

        X = check_array(X)

        self.classes_, y = np.unique(y, return_inverse=True)

        if np.nan in self.classes_:
            raise Exception("NaN not supported in class values")

        # fuzzify data
        # p, X_fuzzy, mu_s = fuzzify_mean(X)

        # agreeing not set, require all features to be in agreement
        if self.n_features is None:
            self.n_features = X.shape[1]

        if self.n_features > X.shape[1]:
            self.n_features = X.shape[1]
            # raise Exception("n_features must be <= number features in X")

        # build membership functions for each feature for each class
        self.protos_ = {}
        for class_idx, class_value in enumerate(self.classes_):
            X_class = X[y == class_idx]

            # create protos from n_protos most agreeing
            self.protos_[class_idx] = \
              build_for_class_multi(X_class, self.max_samples,
                                    self.n_features, rs, self.membership_factory,
                                    self.n_protos)

        return self

    def predict(self, X):
        """

        Predict outputs given examples.

        Parameters:
        -----------

        X : the examples to predict (array or matrix)

        Returns:
        --------

        y_pred : Predicted values for each row in matrix.

        """
        if self.protos_ is None:
            raise Exception("Prototypes not initialized. Perform a fit first.")

        X = check_array(X)

        # predict
        return _predict_multi(self.protos_, self.aggregation, self.classes_, X, self.n_features)
