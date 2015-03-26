# -*- coding: utf-8 -*-
"""Fuzzy pattern classifier with genetic algorithm based methods

The module structure is the following:

- The "MultimodalEvolutionaryClassifier" contains the classifier implementing [1].

- The "EnsembleMultimodalEvolutionaryClassifier" contains an emsemble based classifier
  extended from [1] where more than one prototype is allowed per class, see [2].

References:

[1] Stoean, Stoean, Preuss and Dumitrescu, 2005.
[2] Davidsen, Sreedevi, 2015.
  
"""

import logging
import numpy as np
from numpy.random import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_arrays, array2d
from sklearn.neighbors import DistanceMetric
from fylearn.ga import GeneticAlgorithm, helper_n_generations, helper_fitness

logger = logging.getLogger("garules")

def stoean_f(X):
    return StoeanDistance(np.nanmax(X, 0) - np.nanmin(X, 0))

def distancemetric_f(name, **kwargs):
    def _distancemetric_factory(X):
        return DistanceMetric.get_metric(name)
    return _distancemetric_factory

class StoeanDistance(DistanceMetric):
    def __init__(self, d):
        self.d = d
    def pairwise(self, X, Y=None):
        if Y is None:
            Y = X
        R = np.zeros((len(X), len(Y)))
        for idx, x in enumerate(X):
            R[idx,:] = np.sum(np.abs(Y - x) / self.d, 1)
        return R

class MultimodalEvolutionaryClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iterations=10, df=stoean_f):
        self.n_iterations = n_iterations
        self.df = df

    def get_params(self, deep=False):
        return {"n_iterations": self.n_iterations,
                "df": self.df}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def distance_sum(self, X, Y):
        return np.sum(self.distance_.pairwise(X, Y), 1)

    def build_for_class(self, X):

        distance_fitness = lambda P: self.distance_sum(P, X)

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

        # construct distance measure
        self.distance_ = self.df(X)

        # build models
        models = np.zeros((len(self.classes_), X.shape[1]))
        for c_idx, c_value in enumerate(self.classes_):
            models[c_idx,:] = self.build_for_class(X[y == c_value])

        self.models_ = models

        return self

    def predict(self, X):
        X = array2d(X)

        # calculate similarity for the inputs
        R = self.distance_.pairwise(X, self.models_)
        
        # reduce by taking the one with minimum distance
        return self.classes_.take(np.argmin(R, 1))


class EnsembleMultimodalEvolutionaryClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iterations=10, n_models=3, random_state=None):
        self.n_iterations = n_iterations
        self.n_models = n_models
        self.random_state = random_state

    def get_params(self, deep=False):
        return {"n_iterations": self.n_iterations,
                "n_models"    : self.n_models,
                "random_state": self.random_state}

    def set_params(self, **kwargs):
        for key, value in params.items():
            self.setattr(key, value)
        return self

    def build_for_class(self, rs, X):
        
        # distance based fitness function
        distance_fitness = lambda c: np.sum(np.abs(X - c))
        
        # setup GA
        ga = GeneticAlgorithm(fitness_function=helper_fitness(distance_fitness),
                              elitism=3,
                              n_chromosomes=100,
                              n_genes=X.shape[1],
                              p_mutation=0.3,
                              random_state=rs)

        ga = helper_n_generations(ga, self.n_iterations) # advance the GA

        # return the best found parameters for this class.
        chromosomes, fitness = ga.best(1)
        return chromosomes[0]
        
    def fit(self, X, y):
        X = array2d(X)
        X, y = check_arrays(X, y)

        if self.random_state is None:
            random_state = RandomState()
        else:
            random_state = RandomState(self.random_state)

        self.classes_, y_reverse = np.unique(y, return_inverse=True)

        # build models
        models = {}
        for c_idx, c_value in enumerate(self.classes_):
            X_class = X[y == c_value]
            c_models = []
            for i in range(self.n_models):
                # resample
                X_sample = X_class[random_state.choice(len(X_class), len(X_class))]
                c_models.append(self.build_for_class(random_state, X_sample))
            models[c_value] = np.array(c_models)

        self.models_ = models

        return self

    def predict(self, X):
        X = array2d(X)

        M = np.zeros((len(X), len(self.classes_)))
        R = np.zeros((len(X), self.n_models))

        # calculate similarity for the inputs
        for c_idx, c_value in enumerate(self.classes_):
            for m_idx, model in enumerate(self.models_[c_value]):
                R[:,m_idx] = np.sum(np.abs(X - model), 1)
            M[:,c_idx] = np.min(R, 1)

        # reduce by taking the one with minimum distance
        return self.classes_.take(np.argmin(M, 1))
