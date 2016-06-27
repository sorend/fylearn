# -*- coding: utf-8 -*-
"""Fuzzy pattern classifier with genetic algorithm based methods

The module structure is the following:

- The "MultimodalEvolutionaryClassifier" contains the classifier implementing [1].

- The "EnsembleMultimodalEvolutionaryClassifier" contains an emsemble based classifier
  extended from [1] where more than one prototype is allowed per class, see [2].

References:
[1] Stoean, Stoean, Preuss and Dumitrescu, 2005.
[2] Davidsen, Padmavathamma, 2015.
"""

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
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
            R[idx, :] = np.sum(np.abs(Y - x) / self.d, 1)
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

        ga = helper_n_generations(ga, self.n_iterations)  # advance the GA

        # return the best found parameters for this class.
        chromosomes, fitness = ga.best(1)
        return chromosomes[0]

    def fit(self, X, y):
        X = check_array(X)

        self.classes_, y_reverse = np.unique(y, return_inverse=True)

        # construct distance measure
        self.distance_ = self.df(X)

        # build models
        models = np.zeros((len(self.classes_), X.shape[1]))
        for c_idx, c_value in enumerate(self.classes_):
            models[c_idx, :] = self.build_for_class(X[y == c_value])

        self.models_ = models

        return self

    def predict_(self, X):
        X = check_array(X)
        # calculate similarity for the inputs
        return self.distance_.pairwise(X, self.models_)

    def predict(self, X):
        R = self.predict_(X)
        # reduce by taking the one with minimum distance
        return self.classes_.take(np.argmin(R, 1))

    def predict_proba(self, X):
        R = self.predict_(X)
        return 1.0 - normalize(R, 'l1')

class EnsembleMultimodalEvolutionaryClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iterations=10, n_models=3, random_state=None, sample_size=10, n_iterations_weights=10):
        self.n_iterations = n_iterations
        self.n_models = n_models
        self.random_state = random_state
        self.sample_size = sample_size
        self.n_iterations_weights = n_iterations_weights

    def get_params(self, deep=False):
        return {"n_iterations": self.n_iterations,
                "n_models": self.n_models,
                "random_state": self.random_state,
                "sample_size": self.sample_size,
                "n_iterations_weights": self.n_iterations_weights}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.setattr(key, value)
        return self

    def build_for_class(self, rs, X):

        def distance_fitness(c):
            return np.sum(np.abs(X - c))

        # setup GA
        ga = GeneticAlgorithm(fitness_function=helper_fitness(distance_fitness),
                              elitism=3,
                              n_chromosomes=100,
                              n_genes=X.shape[1],
                              p_mutation=0.3,
                              random_state=rs)

        ga = helper_n_generations(ga, self.n_iterations)  # advance the GA

        # return the best found parameters for this class.
        chromosomes, fitness = ga.best(1)
        return chromosomes[0]

    def fit_weights(self, rs, models, X, y):

        n_genes = self.n_models * len(self.classes_)

        def fitness_function(c):
            M = self.predict_(X, models, c)
            y_pred = np.argmin(M, 1)
            return 1.0 - accuracy_score(y, y_pred)

        ga = GeneticAlgorithm(fitness_function=helper_fitness(fitness_function),
                              elitism=3,
                              n_chromosomes=100,
                              n_genes=n_genes,
                              p_mutation=0.3,
                              random_state=rs)

        ga = helper_n_generations(ga, self.n_iterations_weights)  # advance the GA

        chromosomes, fitness = ga.best(1)

        return chromosomes[0]

    def fit(self, X, y):
        X = check_array(X)

        random_state = check_random_state(self.random_state)

        self.classes_, y_reverse = np.unique(y, return_inverse=True)

        if np.nan in self.classes_:
            raise ValueError("NaN class not supported.")

        # build models
        models = {}
        for c_idx, c_value in enumerate(self.classes_):
            X_class = X[y == c_value]
            a_sample_size = min(len(X_class), self.sample_size)
            c_models = []
            for i in range(self.n_models):
                # resample
                X_sample = X_class[random_state.choice(len(X_class), a_sample_size)]
                c_models.append(self.build_for_class(random_state, X_sample))
            models[c_value] = np.array(c_models)

        weights = self.fit_weights(random_state, models, X, y_reverse)

        self.models_ = models
        self.weights_ = weights

        return self

    def predict_(self, X, models, weights):
        X = check_array(X)

        M = np.zeros((len(X), len(self.classes_)))
        R = np.zeros((len(X), self.n_models))

        # calculate similarity for the inputs
        for c_idx, c_value in enumerate(self.classes_):
            for m_idx, model in enumerate(models[c_value]):
                R[:, m_idx] = np.sum(np.abs(X - model), 1)
            M[:, c_idx] = weights[c_idx] * np.sum(R, 1)

        return M

    def predict(self, X):

        M = self.predict_(X, self.models_, self.weights_)

        # reduce by taking the one with minimum distance
        return self.classes_.take(np.argmin(M, 1))

    def predict_proba(self, X):

        M = self.predict_(X, self.models_, self.weights_)

        return 1.0 - normalize(M, 'l1')
