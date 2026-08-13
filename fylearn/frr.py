"""Fuzzy reduction rule based methods

The module structure is the following:

- The "FuzzyReductionRuleClassifier" implements the model learning using the [1] algorithm.
- The "ModifiedFuzzyPatternClassifier" implements the model learning using the [2] algorithm.

References:

[1] S. K. Meher, "A new fuzzy supervised classification method based on aggregation operator,"
    In Proc. 3rd IEEE Conf on Signal-Image tech and internet-based syst, 2007.
    url: https://ieeexplore.ieee.org/document/4618866

[2] U. Monks, V. Lohweg, and Larsen, "Aggregation operator based fuzzy pattern classifier design,"
    In Proc. Conf machine learning in real-time applications, 2009.
    url: https://www.researchgate.net/publication/229035282_Aggregation_Operator_Based_Fuzzy_Pattern_Classifier_Design
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array

import fylearn.fuzzylogic as fl
from fylearn._validation import has_nan_classes

#
# Authors: Søren Atmakuri Davidsen <sorend@gmail.com>
#


def pi_factory(*args: float) -> fl.PiSet:
    return fl.PiSet(p=args[0], r=args[1], q=args[2], m=2.0)


def build_memberships(X: np.ndarray, factory: Callable[..., fl.PiSet]) -> list[fl.PiSet]:
    mins = np.nanmin(X, 0)
    maxs = np.nanmax(X, 0)
    means = np.nanmean(X, 0)
    return [
        factory(means[i] - ((maxs[i] - mins[i]) / 2.0), means[i], means[i] + ((maxs[i] - mins[i]) / 2.0))
        for i in range(len(X.T))
    ]


class FuzzyReductionRuleClassifier(BaseEstimator, ClassifierMixin):
    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {"aggregation": self.aggregation, "membership_factory": self.membership_factory}

    def set_params(self, **kwargs: Any) -> "FuzzyReductionRuleClassifier":
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def __init__(
        self,
        aggregation: Callable[..., Any] = np.mean,
        membership_factory: Callable[..., fl.PiSet] = pi_factory,
    ):
        self.aggregation = aggregation
        self.membership_factory = membership_factory

    def fit(self, X: Any, y: Any) -> "FuzzyReductionRuleClassifier":
        X = check_array(X)

        self.classes_, y = np.unique(y, return_inverse=True)

        if has_nan_classes(self.classes_):
            raise Exception("nan not supported for class values")

        # build membership functions for each feature for each class
        self.protos_: dict[int, list[fl.PiSet]] = {}
        for class_idx, class_value in enumerate(self.classes_):
            self.protos_[class_idx] = build_memberships(X[y == class_idx], self.membership_factory)

        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "protos_"):
            raise Exception("Prototypes not initialized. Perform a fit first.")

        X = check_array(X)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        R = np.zeros((n_samples, n_classes))

        for class_idx in range(n_classes):
            # Protos for this class
            protos = self.protos_[class_idx]

            # Calculate membership for all samples and all features for this class
            # M will be (n_samples, n_features)
            M = np.zeros((n_samples, n_features))
            for j in range(n_features):
                M[:, j] = protos[j](X[:, j])

            # Aggregate over features (axis 1)
            try:
                # Try to use axis parameter if supported (e.g. numpy functions, OWA)
                R[:, class_idx] = self.aggregation(M, axis=1)
            except TypeError:
                # Fallback to apply_along_axis for generic functions
                R[:, class_idx] = np.apply_along_axis(self.aggregation, 1, M)

        return self.classes_.take(np.argmax(R, axis=1))


def build_aiwa_operator(andness: float, m: int) -> fl.AndnessDirectedAveraging:
    return fl.aa(andness)


def build_owa_operator(andness: float, m: int) -> fl.OWA:
    beta = andness / (1.0 - andness)
    v = np.array(range(m)) + 1.0
    w = ((v / m) ** beta) - (((v - 1.0) / m) ** beta)
    return fl.owa(w)


class ModifiedFuzzyPatternClassifier(BaseEstimator, ClassifierMixin):
    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {"D": self.D, "pce": self.pce, "andness": self.andness, "operator": self.operator}

    def set_params(self, **kwargs: Any) -> "ModifiedFuzzyPatternClassifier":
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def __init__(self, D: int = 2, pce: float = 0.0, andness: float = 0.75, operator: str = "aiwa"):
        if D not in (2, 4, 6, 8):
            raise ValueError("D must be in {2, 4, 6, 8}")

        if pce < 0.0 or pce > 1.0:
            raise ValueError("pce must be within [0, 1]")

        if andness < 0.5 or andness > 1.0:
            raise ValueError("andness must be within [0.5, 1]")

        if operator not in ("aiwa", "owa"):
            raise ValueError("operator must be 'aiwa' or 'owa'")

        self.D = D
        self.pce = pce
        self.andness = andness
        self.operator = operator

    def fit(self, X: Any, y: Any) -> "ModifiedFuzzyPatternClassifier":
        self.classes_ = np.unique(y)

        self.S_: list[np.ndarray] = []
        self.C_: list[np.ndarray] = []

        # learn mu function parameters
        for idx, clz in enumerate(self.classes_):
            m_max = np.max(X[y == clz], 0)
            m_min = np.min(X[y == clz], 0)
            delta = np.maximum((m_max - m_min) / 2.0, 0.0001)
            self.S_.append(delta + m_min)
            self.C_.append((1.0 + (2.0 * self.pce)) * delta)

        # construct aggregation operator
        self.operator_ = globals()["build_" + self.operator + "_operator"](self.andness, X.shape[1])

        return self

    def predict(self, X: Any) -> np.ndarray:
        def mu_mfpc(m: np.ndarray, S: np.ndarray, C: np.ndarray) -> np.ndarray:
            return 2 ** -((np.abs(m - S) / C) ** self.D)

        R = np.zeros((len(X), len(self.classes_)))
        for idx, clz in enumerate(self.classes_):
            R[:, idx] = self.operator_(mu_mfpc(X, self.S_[idx], self.C_[idx]))

        return self.classes_.take(np.argmax(R, -1))
