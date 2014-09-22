# -*- coding: utf-8 -*-
"""Fuzzy reduction rule based methods

The module structure is the following:

- The "FuzzyReductionRuleClassifier" implements the model learning using the
  [1, 2] algorithm.

References:

[1] Meher, 2009.
  
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_arrays, column_or_1d
import fylearn.fuzzylogic as fl

#
# Authors: Søren Atmakuri Davidsen <sorend@gmail.com>
#


def pi_factory(fuzzifier=2.0):
    def factory(a, b, c):
        return fl.pi(a, b, c, fuzzifier)
    return factory

def build_memberships(X, factory):
    mins  = np.nanmin(X, 0)
    maxs  = np.nanmax(X, 0)
    means = np.nanmean(X, 0)
    return [ factory(mins[i], means[i], maxs[i]) for i in range(len(X.T)) ]

class FuzzyReductionRuleClassifier(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {"aggregation": self.aggregation,
                "membership_factory": self.membership_factory}

    def set_params(self, **kwargs):
        for key, value in params.items():
            self.setattr(key, value)
        return self
    
    def __init__(self, aggregation=np.mean, membership_factory=pi_factory(2.0)):
        self.aggregation = aggregation
        self.membership_factory = membership_factory
    
    def fit(self, X, y):

        X, = check_arrays(X)

        self.classes_, y = np.unique(y, return_inverse=True)

        if np.nan in self.classes_:
            raise "nan not supported for class values"

        # build membership functions for each feature for each class
        self.protos_ = {}
        for class_idx, class_value in enumerate(self.classes_):
            self.protos_[class_idx] = build_memberships(X[y == class_idx], self.membership_factory)

        return self

    def predict(self, X):

        if self.protos_ is None:
            raise Exception("Prototypes not initialized. Perform a fit first.")
        
        X, = check_arrays(X)

        def predict_one(x):
            R = np.zeros(len(self.protos_))
            for class_idx, proto in self.protos_.items():
                M = [ proto[i](x[i]) for i in range(len(x)) if np.isfinite(x[i]) ]
                R[class_idx] = self.aggregation(M)
            return self.classes_.take(np.argmax(R))

        # predict the lot
        return np.apply_along_axis(predict_one, 1, X)
