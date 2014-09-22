# -*- coding: utf-8 -*-
"""Random agreement ensemble Fuzzy reduction rule based method.

The module structure is the following:

- The "RandomAgreementEnsembleFRRClassifier" implements the model learning using the
  [1, 2] algorithm.

References:

[1] Davidsen, 2014.
  
"""
import numpy as np
import scipy.stats as stats

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_arrays, column_or_1d

import fylearn.fuzzylogic as fl
from fylearn.frr import pi_factory

def static_one(x):
    return 1.0

def build_memberships(S, A, factory):
    mins  = np.nanmin(S, 0)
    maxs  = np.nanmax(S, 0)
    means = np.nanmean(S, 0)

    def func_or_one(i):
        if A[i]:
            return factory(mins[i], means[i], maxs[i])
        else:
            return static_one
    
    return [ func_or_one(i) for i in range(S.shape[1]) ]

def agreement_t_test(a, b):
    """ Check agreement based on means of two samples, using the t-statistic. """
    means1, stds1 = np.nanmean(a, 0), np.nanstd(a, 0)
    means2, stds2 = np.nanmean(b, 0), np.nanstd(b, 0)

    t1, p1 = stats.ttest_1samp(b, means1)
    # t2, p2 = stats.ttest_1samp(a, means2)

    # select agreeing featurs (p <= 0.05)
    return p1 < 0.05

class RandomAgreementEnsemblesFRRClassifier(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {"aggregation": self.aggregation,
                "n_samples": self.n_samples,
                "sample_length": self.sample_length,
                "membership_factory": self.membership_factory,
                "random_seed": self.random_seed}

    def set_params(self, **kwargs):
        for key, value in params.items():
            self.setattr(key, value)
        return self
    
    def __init__(self, n_samples=10, sample_length=100,
                 aggregation=np.mean, membership_factory=pi_factory(2.0),
                 random_seed=None):
        """
        Initialize the classifier

        Parameters:
        -----------
        n_samples : the number of samples to draw for finding agreement

        sample_length : the length of each sample. If <1.0 then treat as a percentage.
                        If >=1.0 then treat as an absolute number.
        
        """
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.aggregation = aggregation
        self.membership_factory = membership_factory
        self.random_seed = random_seed
    
    def fit(self, X, y):

        # set the random seed
        if self.random_seed is not None:
            np.random.seed = self.random_seed

        X, = check_arrays(X)
        m = len(X)

        self.classes_, y = np.unique(y, return_inverse=True)

        if np.nan in self.classes_:
            raise "nan not supported for class values"

        # build membership functions for each feature for each class
        self.protos_ = {}
        for class_idx, class_value in enumerate(self.classes_):
            X_class = X[y == class_idx]
            self.protos_[class_idx] = []
            for sample in [ X[np.random.permutation(m)[:self.n_samples * 2]] for x in range(self.sample_length) ]:
                sample1, sample2 = sample[len(sample) / 2:], sample[:len(sample) / 2]
                agreement = agreement_t_test(sample1, sample2)

                if np.any(agreement == True):
                    self.protos_[class_idx].append(build_memberships(sample,
                                                                     agreement,
                                                                     self.membership_factory))
        return self

    def predict(self, X):

        if self.protos_ is None:
            raise Exception("Prototypes not initialized. Perform a fit first.")

        X, = check_arrays(X)

        def predict_one(x):
            R = np.zeros(len(self.protos_))
            for class_idx, proto in self.protos_.items():
                M = []
                for p in proto:
                    for i in range(len(x)):
                        M.append(self.aggregation([ p_f(x[i]) for p_f in p ]))
                print "M", M
                R[class_idx] = np.mean(M)

            return self.classes_.take(np.argmax(R))

        # predict the lot
        return np.apply_along_axis(predict_one, 1, X)

