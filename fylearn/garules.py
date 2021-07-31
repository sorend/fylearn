# -*- coding: utf-8 -*-
"""Fuzzy pattern classifier with genetic algorithm based methods.

The module structure is the following:

- The "MultimodalEvolutionaryClassifier" contains the classifier implementing [1].

- The "EnsembleMultimodalEvolutionaryClassifier" contains an emsemble based classifier
  extended from [1] where more than one prototype is allowed per class, see [2].

### References:

[1] C. Stoean, R. Stoean, M. Preuss and D. Dumitrescu, "Diabetes diagnosis through the
    means of multi-modal evolutionary algorithm," In Proc. 1st East Euro. Conf. on Health
    Care Modelling and Computation, pages 277-289, 2005.

[2] S. A. Davidsen, and M. Padmavathamma, "Multi-modal evolutionary ensemble
    classification in medical diagnosis problems," In Proc. Recent Advances in Medical
    Informatics, Kochi, 2015.
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
    """Distance measure used by Stoean."""
    return StoeanDistance(np.nanmax(X, 0) - np.nanmin(X, 0))


def distancemetric_f(name, **kwargs):
    """Return a factory for using sklearns DistanceMetrics."""
    def _distancemetric_factory(X):
        return DistanceMetric.get_metric(name)
    return _distancemetric_factory


class StoeanDistance(DistanceMetric):
    """Stoeans distance metric."""

    def __init__(self, d):
        """Initialize."""
        self.d = d

    def pairwise(self, X, Y=None):
        """Calculate pairwise distances."""
        if Y is None:
            Y = X
        R = np.zeros((len(X), len(Y)))
        for idx, x in enumerate(X):
            R[idx, :] = np.sum(np.abs(Y - x) / self.d, 1)
        return R


class MultimodalEvolutionaryClassifier(BaseEstimator, ClassifierMixin):
    """Multi-modal evolutionary classifier learns a reference vector for each class using a GA optimiser."""

    def __init__(self, n_iterations=10, df=stoean_f, random_state=None):
        """Initialize."""
        self.n_iterations = n_iterations
        self.df = df
        self.random_state = check_random_state(random_state)

    def get_params(self, deep=False):
        """Return parameter values of classifier."""
        return {"n_iterations": self.n_iterations,
                "random_state": self.random_state,
                "df": self.df}

    def set_params(self, **kwargs):
        """Configure classifier from parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def distance_sum(self, X, Y):
        """Calculate sum of pairwise distances."""
        return np.sum(self.distance_.pairwise(X, Y), 1)

    def build_for_class(self, X):
        """Learn parameters for a single class."""
        def distance_fitness(P):
            return self.distance_sum(P, X)

        # setup GA
        ga = GeneticAlgorithm(fitness_function=distance_fitness,
                              elitism=3,
                              n_chromosomes=100,
                              n_genes=X.shape[1],
                              p_mutation=0.3,
                              random_state=self.random_state)

        ga = helper_n_generations(ga, self.n_iterations)  # advance the GA

        # return the best found parameters for this class.
        chromosomes, fitness = ga.best(1)
        return chromosomes[0]

    def fit(self, X, y):
        """Fit classifier given data X and labels y."""
        X = check_array(X)

        self.classes_, _ = np.unique(y, return_inverse=True)

        # construct distance measure
        self.distance_ = self.df(X)

        # build models
        models = np.zeros((len(self.classes_), X.shape[1]))
        for c_idx, c_value in enumerate(self.classes_):
            models[c_idx, :] = self.build_for_class(X[y == c_value])

        self.models_ = models

        return self

    def _predict(self, X):
        X = check_array(X)
        # calculate similarity for the inputs
        return self.distance_.pairwise(X, self.models_)

    def predict(self, X):
        """Predict class for instance X."""
        R = self._predict(X)
        # reduce by taking the one with minimum distance
        return self.classes_.take(np.argmin(R, 1))

    def predict_proba(self, X):
        """Predict class probabilities for instance X."""
        R = self._predict(X)
        return 1.0 - normalize(R, 'l1')


class EnsembleMultimodalEvolutionaryClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble of MEC classifiers."""

    def __init__(self, n_iterations=10, n_models=3, random_state=None, sample_size=10, n_iterations_weights=10):
        """Initialize."""
        self.n_iterations = n_iterations
        self.n_models = n_models
        self.random_state = check_random_state(random_state)
        self.sample_size = sample_size
        self.n_iterations_weights = n_iterations_weights

    def get_params(self, deep=False):
        """Return parameters of classifier."""
        return {"n_iterations": self.n_iterations,
                "n_models": self.n_models,
                "random_state": self.random_state,
                "sample_size": self.sample_size,
                "n_iterations_weights": self.n_iterations_weights}

    def set_params(self, **kwargs):
        """Configure classifier from parameters."""
        for key, value in kwargs.items():
            self.setattr(key, value)
        return self

    def _build_for_class(self, rs, X):
        """Learn parameters for a single class."""

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

    def _fit_weights(self, rs, models, X, y):

        n_genes = self.n_models * len(self.classes_)

        def fitness_function(c):
            M = self._predict(X, models, c)
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
        """Fit classifier to data X and labels y."""
        X = check_array(X)

        random_state = self.random_state

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
                c_models.append(self._build_for_class(random_state, X_sample))
            models[c_value] = np.array(c_models)

        weights = self._fit_weights(random_state, models, X, y_reverse)

        self.models_ = models
        self.weights_ = weights

        return self

    def _predict(self, X, models, weights):
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
        """Predict class for instance X."""
        M = self._predict(X, self.models_, self.weights_)

        # reduce by taking the one with minimum distance
        return self.classes_.take(np.argmin(M, 1))

    def predict_proba(self, X):
        """Predict class probabilities for instance X."""
        M = self._predict(X, self.models_, self.weights_)

        return 1.0 - normalize(M, 'l1')


class DiverseEnsembleMultimodalEvolutionaryClassifier(BaseEstimator, ClassifierMixin):
    """Diverse Ensemble of MEC classifiers."""

    def __init__(self, n_iterations=10, n_models=3, random_state=None, sample_size=10, n_iterations_weights=10):
        """Initialize."""
        self.n_iterations = n_iterations
        self.n_models = n_models
        self.random_state = check_random_state(random_state)
        self.sample_size = sample_size
        self.n_iterations_weights = n_iterations_weights

    def get_params(self, deep=False):
        """Return parameters of classifier."""
        return {"n_iterations": self.n_iterations,
                "n_models": self.n_models,
                "random_state": self.random_state,
                "sample_size": self.sample_size,
                "n_iterations_weights": self.n_iterations_weights}

    def set_params(self, **kwargs):
        """Configure classifier from parameters."""
        for key, value in kwargs.items():
            self.setattr(key, value)
        return self

    def _build_for_class(self, rs, X):
        """Learn parameters for a single class."""

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

    def _fit_weights(self, rs, models, X, y):

        n_genes = self.n_models * len(self.classes_)

        def single_fitness_function(c, P):

            D = np.mean(P, 0)
            d = np.sum(np.abs(c - D))

            M = self._predict(X, models, c)
            y_pred = np.argmin(M, 1)
            return d * (1.0 - accuracy_score(y, y_pred))

        def fitness_function(population):
            res = []
            for row in population:
                res.append(single_fitness_function(row, population))
            return np.array(res)

        ga = GeneticAlgorithm(fitness_function=fitness_function,
                              elitism=3,
                              n_chromosomes=100,
                              n_genes=n_genes,
                              p_mutation=0.3,
                              random_state=rs)

        ga = helper_n_generations(ga, self.n_iterations_weights)  # advance the GA

        chromosomes, fitness = ga.best(1)

        return chromosomes[0]

    def fit(self, X, y):
        """Fit classifier to data X and labels y."""
        X = check_array(X)

        random_state = self.random_state

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
                c_models.append(self._build_for_class(random_state, X_sample))
            models[c_value] = np.array(c_models)

        weights = self._fit_weights(random_state, models, X, y_reverse)

        self.models_ = models
        self.weights_ = weights

        return self

    def _predict(self, X, models, weights):
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
        """Predict class for instance X."""
        M = self._predict(X, self.models_, self.weights_)

        # reduce by taking the one with minimum distance
        return self.classes_.take(np.argmin(M, 1))

    def predict_proba(self, X):
        """Predict class probabilities for instance X."""
        M = self._predict(X, self.models_, self.weights_)

        return 1.0 - normalize(M, 'l1')
