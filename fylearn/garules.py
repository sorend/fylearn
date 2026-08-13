"""Fuzzy pattern classifier with genetic algorithm based methods

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
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import DistanceMetric, accuracy_score
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from fylearn._validation import has_nan_classes
from fylearn.ga import GeneticAlgorithm, helper_fitness, helper_n_generations

logger = logging.getLogger("garules")

def stoean_f(X: np.ndarray) -> "StoeanDistance":
    return StoeanDistance(np.nanmax(X, 0) - np.nanmin(X, 0))

def distancemetric_f(name: str, **kwargs: Any) -> Callable[[np.ndarray], DistanceMetric]:
    def _distancemetric_factory(X: np.ndarray) -> DistanceMetric:
        return DistanceMetric.get_metric(name)
    return _distancemetric_factory

class StoeanDistance(DistanceMetric):
    def __init__(self, d: np.ndarray):
        self.d = d

    def pairwise(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        R = np.zeros((len(X), len(Y)))
        for idx, x in enumerate(X):
            R[idx, :] = np.sum(np.abs(Y - x) / self.d, 1)
        return R

class MultimodalEvolutionaryClassifier(BaseEstimator, ClassifierMixin):
    """Multi-modal evolutionary classifier learns a reference vector for each class using a GA optimiser.

    """

    def __init__(self, n_iterations: int = 10, df: Callable = stoean_f, random_state: Any = None):
        self.n_iterations = n_iterations
        self.df = df
        self.random_state = check_random_state(random_state)

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {"n_iterations": self.n_iterations,
                "random_state": self.random_state,
                "df": self.df}

    def set_params(self, **kwargs: Any) -> "MultimodalEvolutionaryClassifier":
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def distance_sum(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.sum(self.distance_.pairwise(X, Y), 1)

    def build_for_class(self, X: np.ndarray) -> np.ndarray:

        def distance_fitness(P: np.ndarray) -> np.ndarray:
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

    def fit(self, X: Any, y: Any) -> "MultimodalEvolutionaryClassifier":
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

    def predict_(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        # calculate similarity for the inputs
        return self.distance_.pairwise(X, self.models_)

    def predict(self, X: Any) -> np.ndarray:
        R = self.predict_(X)
        # reduce by taking the one with minimum distance
        return self.classes_.take(np.argmin(R, 1))

    def predict_proba(self, X: Any) -> np.ndarray:
        R = self.predict_(X)
        return 1.0 - normalize(R, 'l1')

class EnsembleMultimodalEvolutionaryClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_iterations: int = 10,
        n_models: int = 3,
        random_state: Any = None,
        sample_size: int = 10,
        n_iterations_weights: int = 10,
    ):
        self.n_iterations = n_iterations
        self.n_models = n_models
        self.random_state = check_random_state(random_state)
        self.sample_size = sample_size
        self.n_iterations_weights = n_iterations_weights

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {"n_iterations": self.n_iterations,
                "n_models": self.n_models,
                "random_state": self.random_state,
                "sample_size": self.sample_size,
                "n_iterations_weights": self.n_iterations_weights}

    def set_params(self, **kwargs: Any) -> "EnsembleMultimodalEvolutionaryClassifier":
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def build_for_class(self, rs: Any, X: np.ndarray) -> np.ndarray:

        def distance_fitness(c: np.ndarray) -> float:
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

    def fit_weights(self, rs: Any, models: dict[Any, np.ndarray], X: np.ndarray, y: np.ndarray) -> np.ndarray:

        n_genes = self.n_models * len(self.classes_)

        def fitness_function(c: np.ndarray) -> float:
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

    def fit(self, X: Any, y: Any) -> "EnsembleMultimodalEvolutionaryClassifier":
        X = check_array(X)

        random_state = self.random_state

        self.classes_, y_reverse = np.unique(y, return_inverse=True)

        if has_nan_classes(self.classes_):
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

    def predict_(self, X: np.ndarray, models: dict[Any, np.ndarray], weights: np.ndarray) -> np.ndarray:
        X = check_array(X)

        M = np.zeros((len(X), len(self.classes_)))
        R = np.zeros((len(X), self.n_models))

        # calculate similarity for the inputs
        for c_idx, c_value in enumerate(self.classes_):
            for m_idx, model in enumerate(models[c_value]):
                R[:, m_idx] = np.sum(np.abs(X - model), 1)
            # per-model weights for this class
            w = weights[c_idx * self.n_models : (c_idx + 1) * self.n_models]
            M[:, c_idx] = np.sum(w * R, 1)

        return M

    def predict(self, X: Any) -> np.ndarray:

        M = self.predict_(X, self.models_, self.weights_)

        # reduce by taking the one with minimum distance
        return self.classes_.take(np.argmin(M, 1))

    def predict_proba(self, X: Any) -> np.ndarray:

        M = self.predict_(X, self.models_, self.weights_)

        return 1.0 - normalize(M, 'l1')
