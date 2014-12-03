# -*- coding: utf-8 -*-
"""Random agreement Fuzzy Pattern Classifier method.

The module structure is the following:

- The "RandomAgreementFuzzyPatternClassifier" implements the model learning using the [1] algorithm.

References:

[1] Davidsen, 2014.
  
"""
import logging
import numpy as np
from numpy.random import RandomState
import scipy.stats as stats

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_arrays, column_or_1d, array2d, check_random_state

import fylearn.fuzzylogic as fl

def agreement_t_test(a, b):
    """ Check agreement based on means of two samples, using the t-statistic. """
    means1, stds1 = np.nanmean(a, 0), np.nanstd(a, 0)
    means2, stds2 = np.nanmean(b, 0), np.nanstd(b, 0)

    t1, p1 = stats.ttest_1samp(b, means1)
    # t2, p2 = stats.ttest_1samp(a, means2)

    # select agreeing featurs (p <= 0.05)
    return p1 < 0.05

def fuzzify(A):
    # output for fuzzified values
    R = np.zeros((A.shape[0], A.shape[1] * 3))

    cmin, cmax, cmean = np.nanmin(A, 0), np.nanmax(A, 0), np.nanmean(A, 0)

    left = np.array([cmin - (cmax - cmin), cmin, cmax]).T
    middle = np.array([cmin, cmean, cmax]).T
    right = np.array([cmin, cmax, cmax + (cmax - cmin)]).T

    for i in range(A.shape[1]):
        f_l = fl.TriangularSet(*left[i])
        f_m = fl.TriangularSet(*middle[i])
        f_r = fl.TriangularSet(*right[i])
        R[:,(i*3)] = f_l(A[:,i])
        R[:,(i*3)+1] = f_m(A[:,i])
        R[:,(i*3)+2] = f_r(A[:,i])

    return R

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

def triangular_factory(*args):
    return fl.TriangularSet(args[0], args[1], args[2])
                
def build_memberships(X, idxs, factory):
    mins  = np.nanmin(X, 0)
    maxs  = np.nanmax(X, 0)
    means = np.nanmean(X, 0)
    return [ (i, factory(means[i] - ((maxs[i] - mins[i]) / 2.0), means[i], means[i] + ((maxs[i] - mins[i]) / 2.0))) for i in idxs ]

logger = logging.getLogger("rafpc")

class RandomAgreementFuzzyPatternClassifier(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {"n_protos": self.n_protos,
                "n_features": self.n_features,
                "n_samples": self.n_samples,
                "sample_length": self.sample_length,
                "aggregation": self.aggregation,
                "membership_factory": self.membership_factory,
                "random_state": self.random_state}

    def set_params(self, **kwargs):
        for key, value in params.items():
            self.setattr(key, value)
        return self
    
    def __init__(self, n_protos=5, n_features=None,
                 n_samples=100, sample_length=10,
                 aggregation=fl.mean, membership_factory=triangular_factory,
                 random_state=None):
        """
        Initialize the classifier

        Parameters:
        -----------
        n_protos : the number of prototypes to keep for each class.

        n_features : the number of features to include in each prototype.
                     None means use all features.
        
        n_samples : the number of samples to draw in finding agreement.

        sample_length : the number of elements in each sample.

        aggregation : The aggregation to use for inference.

        membership_factory : the factory to create membership functions.

        random_state : The random state to use for drawing samples.
                       None means no specific random state.
        
        """
        self.n_protos = n_protos
        self.n_features = n_features
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.aggregation = aggregation
        self.membership_factory = membership_factory
        self.random_state = random_state

    def fit(self, X, y):

        # get random
        rs = RandomState(self.random_state)

        X = array2d(X)
        
        X, y = check_arrays(X, y)
        n = len(X)

        self.classes_, y = np.unique(y, return_inverse=True)

        if np.nan in self.classes_:
            raise Exception("NaN not supported in class values")

        # agreeing not set, require all features to be in agreement
        if self.n_features is None:
            self.n_features = X.shape[1]

        if self.n_features > X.shape[1]:
            raise Exception("n_features must be <= number features in X")

        # the owa operator to use for aggregation (assign equal weight to all elements)
        owa_weights = [1.0 / self.n_features] * self.n_features
        aa_f = fl.owa(owa_weights)
        
        # build membership functions for each feature for each class
        self.protos_ = {}
        for class_idx, class_value in enumerate(self.classes_):
            X_class = X[y == class_idx]

            agreements = []
            for x in range(self.n_samples):
                # draw the samples (use re-sampling by hashing, if the number of elements is too few)
                sample_idx = rs.permutation(self.sample_length * 2) % len(X_class)
                sample = X_class[sample_idx]
                sample1, sample2 = sample[:self.sample_length], sample[self.sample_length:]
                #
                agreement, d = agreement_fuzzy(aa_f, sample1, sample2)
                agreements.append({"agreement": agreement, "d": d, "sample": sample})

            agreements.sort(key=lambda x: x["agreement"]) # in-place sort list on agreement

            # create protos from n_protos most agreeing
            self.protos_[class_idx] = []
            for a in agreements[-self.n_protos:]:

                # logger.info("d %s" %(str(a["d"]), ))

                ranking = np.argsort(1.0 - d)
                ranking = ranking[:self.n_features]
                
                self.protos_[class_idx].append(build_memberships(a["sample"], ranking, self.membership_factory))
                
        return self

    def predict(self, X):

        if self.protos_ is None:
            raise Exception("Prototypes not initialized. Perform a fit first.")

        X = array2d(X)

        Mus = np.zeros(X.shape)                        # holds output per prototype
        R = np.zeros((X.shape[0], len(self.classes_))) # holds output for each class
        feature_nos = range(self.n_features)         # index for agreements

        #for k, v in self.protos_.items():
        #    logger.info("class %d prototypes %d min-v-len %d" % (k, len(v), min(map(lambda x: len(x), v))))

        # class_idx has class_prototypes membership functions
        for class_idx, class_prototypes in self.protos_.items():
            C = np.zeros((X.shape[0], len(class_prototypes)))
            for j, cp in enumerate(class_prototypes):
                for i in feature_nos:
                    f_idx, f_f = cp[i]
                    Mus[:,i] = f_f(X[:,f_idx])
                C[:,j] = self.aggregation(Mus)
            R[:,class_idx] = np.max(C, 1)

        return self.classes_.take(np.argmax(R, 1))
