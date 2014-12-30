# -*- coding: utf-8 -*-
"""Fuzzy pattern classifier with genetic algorithm based methods

The module structure is the following:

- The "MultimodalEvolutionaryClassifier" contains the classifier implementing [1].

References:

[1] Stoean, Stoean, Preuss and Dumitrescu, 2005.
  
"""

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_arrays, array2d
from fylearn.ga import GeneticAlgorithm, helper_n_generations

logger = logging.getLogger("garules")

class MultimodalEvolutionaryClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iterations=10):
        self.n_iterations = n_iterations

    def get_params(self, deep=False):
        return {"n_iterations": self.n_iterations}

    def set_params(self, **kwargs):
        for key, value in params.items():
            self.setattr(key, value)
        return self

    def build_for_class(self, X):
        
        # distance based fitness function
        distance_fitness = lambda c: np.sum(np.abs(X - c) / self.d)
        
        # setup GA
        ga = GeneticAlgorithm(fitness_function=distance_fitness,
                              elitism=3,
                              n_chromosomes=100,
                              n_genes=X.shape[1],
                              p_mutation=0.3)

        ga = helper_n_generations(ga, self.n_iterations) # advance the GA

        # return the best found parameters for this class.
        chromosomes, fitness = ga.best(1)
        return chromosomes[0]
        
    def fit(self, X, y):
        X = array2d(X)
        X, y = check_arrays(X, y)

        self.classes_, y_reverse = np.unique(y, return_inverse=True)

        # calculate normalization parameter for distance measure
        b = np.nanmax(X, 0) # find b and a (max, min) columnwise.
        a = np.nanmin(X, 0)
        self.d = b - a

        # build models
        models = {}
        for c_idx, c_value in enumerate(self.classes_):
            models[c_value] = self.build_for_class(X[y == c_value])

        self.models_ = models

        return self

    def predict(self, X):
        X = array2d(X)

        R = np.zeros((len(X), len(self.classes_))) # prediction output

        # calculate similarity for the inputs
        for c_idx, c_value in enumerate(self.classes_):
            R[:,c_idx] = np.sum(np.abs(X - self.models_[c_value]) / self.d, 1)
            
        # reduce by taking the one with minimum distance
        return self.classes_.take(np.argmin(R, 1))
