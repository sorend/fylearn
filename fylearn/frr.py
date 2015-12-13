# -*- coding: utf-8 -*-
"""Fuzzy reduction rule based methods

The module structure is the following:

- The "FuzzyReductionRuleClassifier" implements the model learning using the
  [1] algorithm.
- The "ModifiedFuzzyPatternClassifier" implements the model learning using the
  [2] algorithm.

References:

[1] Meher, 2009.
[2] Monks and Larsen, 2008.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
import fylearn.fuzzylogic as fl

#
# Authors: Søren Atmakuri Davidsen <sorend@gmail.com>
#


def pi_factory(*args):
    return fl.PiSet(p=args[0], r=args[1], q=args[2], m=2.0)

def build_memberships(X, factory):
    mins = np.nanmin(X, 0)
    maxs = np.nanmax(X, 0)
    means = np.nanmean(X, 0)
    return [ factory(means[i] - ((maxs[i] - mins[i]) / 2.0),
                     means[i],
                     means[i] + ((maxs[i] - mins[i]) / 2.0))
             for i in range(len(X.T)) ]

class FuzzyReductionRuleClassifier(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {"aggregation": self.aggregation,
                "membership_factory": self.membership_factory}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def __init__(self, aggregation=np.mean, membership_factory=pi_factory):
        self.aggregation = aggregation
        self.membership_factory = membership_factory

    def fit(self, X, y):

        X = check_array(X)

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

        X = check_array(X)

        def predict_one(x):
            R = np.zeros(len(self.protos_))
            for class_idx, proto in self.protos_.items():
                M = [ proto[i](x[i]) for i in range(len(x)) if np.isfinite(x[i]) ]
                R[class_idx] = self.aggregation(M)
            return self.classes_.take(np.argmax(R))

        # predict the lot
        return np.apply_along_axis(predict_one, 1, X)


def build_aiwa_operator(andness, m):
    return fl.aa(andness)

def build_owa_operator(andness, m):
    beta = andness / (1.0 - andness)
    v = np.array(range(m)) + 1.0
    w = ((v / m) ** beta) - (((v - 1.0) / m) ** beta)
    return fl.owa(w)

class ModifiedFuzzyPatternClassifier(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {"D": self.D,
                "pce": self.pce,
                "andness": self.andness,
                "operator": self.operator}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def __init__(self, D=2, pce=0.0, andness=0.75, operator="aiwa"):

        if D not in (2, 4, 6, 8):
            raise ValueError("D must be in {2, 4, 6, 8}")

        if pce < 0.0 or pce > 1.0:
            raise ValueError("pcc must be in [0, 1]")

        if andness < 0.5 or andness > 1.0:
            raise ValueError("andness must be in [0.5, 1]")

        if operator not in ("aiwa", "owa"):
            raise ValueError("operator must be 'aiwa' or 'owa'")

        self.D = D
        self.pce = pce
        self.andness = andness
        self.operator = operator

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        self.S_ = []
        self.C_ = []

        # learn mu function parameters
        for idx, clz in enumerate(self.classes_):
            m_max = np.max(X[y == clz], 0)
            m_min = np.min(X[y == clz], 0)
            delta = (m_max - m_min) / 2.0
            self.S_.append(delta + m_min)
            self.C_.append((1.0 + (2.0 * self.pce)) * delta)

        # construct aggregation operator
        self.operator_ = globals()["build_" + self.operator + "_operator"](self.andness, X.shape[1])

    def predict(self, X):

        def mu_mfpc(m, S, C):
            return 2 ** -((np.abs(m - S) / C) ** self.D)

        R = np.zeros((len(X), len(self.classes_)))
        for idx, clz in enumerate(self.classes_):
            R[:, idx] = self.operator_(mu_mfpc(X, self.S_[idx], self.C_[idx]))

        return self.classes_.take(np.argmax(R, -1))
