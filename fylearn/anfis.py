# -*- coding: utf-8 -*-
"""Adaptive Neuro Fuzzy Inference System

The module structure is the following:

- The "ANFISClassifier" implements a classifier for learning adaptive neuro fuzzy inference system
  (ANFIS)
  [1] algorithm.

References:

[1] ??, ????.
  
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_arrays, column_or_1d
import fylearn.fuzzylogic as fl
from fylearn.frr import build_memberships, pi_factory

#
# Authors: Søren Atmakuri Davidsen <sorend@gmail.com>
#

class ANFISClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
