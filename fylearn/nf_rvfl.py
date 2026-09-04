"""Neuro-Fuzzy Random Vector Functional Link estimators.

The model combines a Gaussian fuzzy layer with a random-vector functional
link hidden layer.  Only the output weights are learned, using regularized
least squares, so fitting is substantially cheaper than back-propagating a
deep network.

References
----------
M. Sajid, A. K. Malik, M. Tanveer, and P. N. Suganthan, "Neuro-Fuzzy
Random Vector Functional Link Neural Network for Classification and
Regression Problems," IEEE Transactions on Fuzzy Systems, vol. 32, no. 5,
pp. 2738-2749, 2024. https://doi.org/10.1109/TFUZZ.2024.3359652

The reference MATLAB implementation is available at
https://github.com/mtanveer1/NeuroFuzzy-RVFL.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .fuzzylogic import p_normalize


class _NeuroFuzzyRVFLBase(BaseEstimator):
    """Shared implementation of the NF-RVFL feature map."""

    # Algorithm adapted from Sajid et al., IEEE T-Fuzzy Syst. 32(5), 2024,
    # doi:10.1109/TFUZZ.2024.3359652.

    def __init__(
        self,
        n_fuzzy_nodes: int = 15,
        n_hidden_nodes: int = 203,
        C: float = 0.001,
        activation: int | str = 5,
        cluster_method: str = "fuzzy_cmeans",
        sigma: float = 1.0,
        cmeans_m: float = 2.0,
        cmeans_iterations: int = 100,
        random_state: Any = None,
    ):
        self.n_fuzzy_nodes = n_fuzzy_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.C = C
        self.activation = activation
        self.cluster_method = cluster_method
        self.sigma = sigma
        self.cmeans_m = cmeans_m
        self.cmeans_iterations = cmeans_iterations
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        if not isinstance(self.n_fuzzy_nodes, (int, np.integer)) or self.n_fuzzy_nodes < 1:
            raise ValueError("n_fuzzy_nodes must be a positive integer")
        if not isinstance(self.n_hidden_nodes, (int, np.integer)) or self.n_hidden_nodes < 1:
            raise ValueError("n_hidden_nodes must be a positive integer")
        if self.C <= 0:
            raise ValueError("C must be greater than zero")
        if self.sigma <= 0:
            raise ValueError("sigma must be greater than zero")
        if self.cmeans_m <= 1:
            raise ValueError("cmeans_m must be greater than one")
        if not isinstance(self.cmeans_iterations, (int, np.integer)) or self.cmeans_iterations < 1:
            raise ValueError("cmeans_iterations must be a positive integer")
        if self.cluster_method not in {"random", "kmeans", "fuzzy_cmeans"}:
            raise ValueError("cluster_method must be 'random', 'kmeans', or 'fuzzy_cmeans'")
        if self.activation not in {1, 2, 3, 4, 5, 6, "sigmoid", "sin", "tribas", "radbas", "tanh", "relu"}:
            raise ValueError("unsupported activation function")

    def _activation_function(self, X: np.ndarray) -> np.ndarray:
        activation = self.activation
        if activation in {1, "sigmoid"}:
            result = np.empty_like(X)
            positive = X >= 0
            result[positive] = 1.0 / (1.0 + np.exp(-X[positive]))
            exp_x = np.exp(X[~positive])
            result[~positive] = exp_x / (1.0 + exp_x)
            return result
        if activation in {2, "sin"}:
            return np.sin(X)
        if activation in {3, "tribas"}:
            return np.maximum(1.0 - np.abs(X), 0.0)
        if activation in {4, "radbas"}:
            return np.exp(-(X**2))
        if activation in {5, "tanh"}:
            return np.tanh(X)
        return np.maximum(X, 0.0)

    def _fuzzy_cmeans(self, X: np.ndarray, random_state: Any) -> np.ndarray:
        """Compute fuzzy C-means centers without a scipy dependency."""
        rng = check_random_state(random_state)
        memberships = rng.random_sample((X.shape[0], self.n_fuzzy_nodes))
        memberships = p_normalize(memberships, axis=1)
        centers = np.empty((self.n_fuzzy_nodes, X.shape[1]))

        for _ in range(self.cmeans_iterations):
            weights = memberships**self.cmeans_m
            centers = (weights.T @ X) / np.maximum(weights.sum(axis=0)[:, None], np.finfo(float).eps)
            distances = np.maximum(self._squared_distances(X, centers), np.finfo(float).eps)
            new_memberships = distances ** (-1.0 / (self.cmeans_m - 1.0))
            new_memberships = p_normalize(new_memberships, axis=1)
            if np.max(np.abs(new_memberships - memberships)) <= 1e-6:
                memberships = new_memberships
                break
            memberships = new_memberships
        return centers

    def _build_centers(self, X: np.ndarray, random_state: Any) -> np.ndarray:
        if self.n_fuzzy_nodes > X.shape[0]:
            raise ValueError("n_fuzzy_nodes cannot exceed the number of training samples")
        if self.cluster_method == "random":
            rng = check_random_state(random_state)
            return X[rng.choice(X.shape[0], self.n_fuzzy_nodes, replace=False)].copy()
        if self.cluster_method == "kmeans":
            return KMeans(
                n_clusters=self.n_fuzzy_nodes,
                n_init=10,
                random_state=random_state,
            ).fit(X).cluster_centers_
        return self._fuzzy_cmeans(X, random_state)

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # This form avoids materializing an (n_samples, n_nodes, n_features)
        # tensor while remaining equivalent to a product of Gaussian MFs.
        distances = (X**2).sum(axis=1, keepdims=True) + (centers**2).sum(axis=1)[None, :]
        distances -= 2.0 * X @ centers.T
        return np.maximum(distances, 0.0)

    def _memberships(self, X: np.ndarray) -> np.ndarray:
        distances = self._squared_distances(X, self.centers_)
        memberships = np.exp(-distances / self.sigma)
        # p_normalize is also the shared fuzzy-logic handling for degenerate
        # rows where all Gaussian memberships underflow to zero.
        return p_normalize(memberships, axis=1)

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        memberships = self._memberships(X)
        fuzzy_output = memberships * (X @ self.alpha_)
        hidden_input = np.column_stack((fuzzy_output, np.full(X.shape[0], 0.1)))
        hidden_output = self._activation_function(hidden_input @ self.weight_hidden_)
        return np.column_stack((fuzzy_output, hidden_output, X))

    def _fit_feature_map(self, X: np.ndarray) -> np.ndarray:
        self._validate_parameters()
        random_state = check_random_state(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.centers_ = self._build_centers(X, random_state)
        self.alpha_ = random_state.random_sample((X.shape[1], self.n_fuzzy_nodes))
        self.weight_hidden_ = random_state.random_sample((self.n_fuzzy_nodes + 1, self.n_hidden_nodes))
        return self._design_matrix(X)

    def _fit_output_weights(self, design: np.ndarray, target: np.ndarray) -> None:
        # C is the paper's regularization parameter.  The resulting ridge
        # penalty is 1/C, as in the published MATLAB implementation.
        penalty = 1.0 / self.C
        n_samples, n_columns = design.shape
        if n_columns <= n_samples:
            system = design.T @ design
            system.flat[:: n_columns + 1] += penalty
            self.beta_ = np.linalg.solve(system, design.T @ target)
        else:
            system = design @ design.T
            system.flat[:: n_samples + 1] += penalty
            self.beta_ = design.T @ np.linalg.solve(system, target)

    def _predict_raw(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("beta_", "centers_", "alpha_", "weight_hidden_"))
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        return self._design_matrix(X) @ self.beta_


class NeuroFuzzyRVFLClassifier(_NeuroFuzzyRVFLBase, ClassifierMixin):
    """Neuro-Fuzzy RVFL classifier for binary and multiclass problems."""

    def fit(self, X: Any, y: Any) -> NeuroFuzzyRVFLClassifier:
        X, y = check_X_y(X, y, dtype=np.float64)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("at least two classes are required")
        target = np.eye(self.classes_.size)[y_encoded]
        design = self._fit_feature_map(X)
        self._fit_output_weights(design, target)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_.take(np.argmax(self._predict_raw(X), axis=1))

    def predict_proba(self, X: Any) -> np.ndarray:
        scores = self._predict_raw(X)
        scores -= np.max(scores, axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


class NeuroFuzzyRVFLRegressor(_NeuroFuzzyRVFLBase, RegressorMixin):
    """Neuro-Fuzzy RVFL regressor supporting single and multiple outputs."""

    def fit(self, X: Any, y: Any) -> NeuroFuzzyRVFLRegressor:
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=True, y_numeric=True)
        self._y_was_1d = y.ndim == 1
        target = y.reshape(-1, 1) if self._y_was_1d else y
        design = self._fit_feature_map(X)
        self._fit_output_weights(design, target)
        self.n_outputs_ = target.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        prediction = self._predict_raw(X)
        return prediction[:, 0] if self._y_was_1d else prediction


# Short aliases match the acronym used in the paper and upstream MATLAB code.
NFRVFLClassifier = NeuroFuzzyRVFLClassifier
NFRVFLRegressor = NeuroFuzzyRVFLRegressor
