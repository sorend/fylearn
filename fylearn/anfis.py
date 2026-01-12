# -*- coding: utf-8 -*-
"""
Adaptive Network-based Fuzzy Inference System (ANFIS) classifier.

This implementation uses heuristic optimization (GA, TLBO, etc.) to tune both
antecedent (membership function) and consequent parameters, avoiding the need
for gradient-based backpropagation.
"""

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.metrics import mean_squared_error
from .fuzzylogic import TriangularSet, prod, p_normalize
from .ga import UnitIntervalGeneticAlgorithm, helper_fitness
from .tlbo import TLBO

logger = logging.getLogger(__name__)


def t_factory(mean, std):
    """
    Factory for TriangularSet initialized with mean and standard deviation.
    """
    return TriangularSet(mean - std, mean, mean + std)


class AnfisClassifier(BaseEstimator, ClassifierMixin):
    """
    ANFIS Classifier optimized via heuristic search (Genetic Algorithms, TLBO, etc.).

    It implements a Takagi-Sugeno fuzzy inference system where:
    - Layer 1: Antecedent membership functions (Triangular).
    - Layer 2: Rule firing strength (Product).
    - Layer 3: Normalization of firing strengths.
    - Layer 4: Consequent linear functions (p*x + q).
    - Layer 5: Weighted sum of consequences.
    """

    def __init__(
        self,
        n_rules=5,
        membership_factory=t_factory,
        optimizer_iterations=100,
        optimizer_pop_size=50,
        random_state=None,
    ):
        """
        Initialize the ANFIS classifier.

        Parameters:
        -----------
        n_rules : int
            Number of fuzzy rules to generate.
        membership_factory : function
            Factory to create membership functions (default: Triangular).
        optimizer_iterations : int
            Number of generations/iterations for the optimizer.
        optimizer_pop_size : int
            Population size for the optimizer.
        random_state : int, RandomState instance or None, optional (default=None)
            The generator used to initialize the optimizer. If int, random_state is the seed used.
        """
        self.n_rules = n_rules
        self.membership_factory = membership_factory
        self.optimizer_iterations = optimizer_iterations
        self.optimizer_pop_size = optimizer_pop_size
        self.random_state = random_state

    def _decode_params(self, params, n_features):
        """
        Decodes a flat parameter vector into antecedent and consequent parts.

        Structure of params:
        [
          rule_0_mf_0_a, rule_0_mf_0_b, rule_0_mf_0_c, ..., (Antecedents: n_rules * n_features * 3)
          rule_0_coeff_0, ..., rule_0_bias, ...              (Consequents: n_rules * (n_features + 1))
        ]
        """
        # 1. Decode Antecedents
        # Each MF (Triangular) has 3 params: a, b, c
        n_antecedent_params = self.n_rules * n_features * 3
        antecedent_flat = params[:n_antecedent_params]

        # Reshape to (n_rules, n_features, 3)
        antecedents = antecedent_flat.reshape(self.n_rules, n_features, 3)

        # Sort a, b, c to ensure valid Triangular sets (a <= b <= c)
        antecedents = np.sort(antecedents, axis=2)

        # 2. Decode Consequents
        # Each rule has (n_features coefficients + 1 bias)
        consequent_flat = params[n_antecedent_params:]
        consequents = consequent_flat.reshape(self.n_rules, n_features + 1)

        return antecedents, consequents

    def _forward_pass(self, X, antecedents, consequents):
        """
        Executes the ANFIS forward pass (Layers 1-5).
        """
        n_samples, n_features = X.shape

        # Layer 1: Fuzzification
        # We need to calculate membership for every sample, every rule, every feature
        # fires: (n_samples, n_rules, n_features)
        fires = np.zeros((n_samples, self.n_rules, n_features))

        for r in range(self.n_rules):
            for f in range(n_features):
                # Create MF object on the fly for evaluation
                # Note: This is slightly inefficient but leverages the existing classes
                params = antecedents[r, f]
                mf = TriangularSet(params[0], params[1], params[2])
                fires[:, r, f] = mf(X[:, f])

        # Layer 2: Rule Firing Strength (Product T-norm)
        # w: (n_samples, n_rules)
        w = prod(fires, axis=2)

        # Layer 3: Normalization
        # w_norm: (n_samples, n_rules)
        w_norm = p_normalize(w, axis=1)

        # Layer 4: Consequent Rules (First Order Takagi-Sugeno)
        # f_out = w_norm * (X * p + q)
        # We need to compute linear output for each rule for each sample

        # Expand X for broadcasting: (n_samples, 1, n_features)
        X_expanded = X[:, np.newaxis, :]

        # Extract coefficients and bias
        coeffs = consequents[:, :-1]  # (n_rules, n_features)
        bias = consequents[:, -1]  # (n_rules,)

        # Linear outputs per rule: (n_samples, n_rules)
        # sum(X * coeffs) + bias
        rule_outputs = np.sum(X_expanded * coeffs, axis=2) + bias

        # Layer 5: Output Summation
        # y_pred = sum(w_norm * rule_outputs)
        y_pred = np.sum(w_norm * rule_outputs, axis=1)

        return y_pred

    def fit(self, X, y):
        """
        Fit the ANFIS model using TLBO optimization.
        """
        X = check_array(X)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)

        n_features = X.shape[1]

        # Calculate parameter bounds
        # Antecedents (a, b, c) should be within feature ranges
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        range_vals = max_vals - min_vals

        # Extend bounds slightly to allow coverage at edges
        lower_bound_ant = np.tile(min_vals - 0.1 * range_vals, (self.n_rules, 3)).flatten()
        upper_bound_ant = np.tile(max_vals + 0.1 * range_vals, (self.n_rules, 3)).flatten()

        # Consequent bounds (coefficients and bias)
        # We initialize them somewhat arbitrarily, e.g., [-10, 10] or relative to target
        # A simpler approach is to let the optimizer find them within a large range
        lower_bound_cons = np.full(self.n_rules * (n_features + 1), -10.0)
        upper_bound_cons = np.full(self.n_rules * (n_features + 1), 10.0)

        lower_bounds = np.concatenate([lower_bound_ant, lower_bound_cons])
        upper_bounds = np.concatenate([upper_bound_ant, upper_bound_cons])

        param_dim = len(lower_bounds)

        # Fitness Function
        def fitness(params):
            # Check for batch processing (TLBO passes a matrix of populations)
            if params.ndim == 2:
                results = np.zeros(params.shape[0])
                for i in range(params.shape[0]):
                    results[i] = fitness(params[i])
                return results

            ant, cons = self._decode_params(params, n_features)
            y_pred_raw = self._forward_pass(X, ant, cons)

            # Use MSE against the integer class labels for training
            # This treats classification as a regression problem (common in ANFIS)
            return mean_squared_error(y_encoded, y_pred_raw)

        # Initialize Optimizer (Using TLBO as it's parameter-free and robust)
        # Note: fylearn.tlbo.TLBO minimizes the function f
        optimizer = TLBO(
            f=fitness,
            lower_bound=lower_bounds,
            upper_bound=upper_bounds,
            n_population=self.optimizer_pop_size,
            random_state=self.random_state,
        )

        # Run Optimization
        # TLBO doesn't have a distinct 'run' method that iterates, we loop manually
        # or use helper if available. Based on nfpc.py, we iterate manually.

        self.history_ = []
        for i in range(self.optimizer_iterations):
            next(optimizer)
            _, best_fitness_list = optimizer.best(1)
            best_fitness = best_fitness_list[0]
            self.history_.append(best_fitness)
            if i % 10 == 0:
                logger.debug("Generation %d: Best RMSE %.4f" % (i, best_fitness))

        # Store best parameters
        best_chrom, _ = optimizer.best(1)
        self.best_params_ = best_chrom[0]

        # Decode and store model state
        self.antecedents_, self.consequents_ = self._decode_params(self.best_params_, n_features)

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        if not hasattr(self, "best_params_"):
            raise Exception("Perform a fit first.")

        X = check_array(X)
        y_pred_raw = self._forward_pass(X, self.antecedents_, self.consequents_)

        # Round to nearest integer class label
        y_pred_idx = np.round(y_pred_raw).astype(int)

        # Clip to valid class indices
        y_pred_idx = np.clip(y_pred_idx, 0, len(self.classes_) - 1)

        return self.classes_.take(y_pred_idx)

    def predict_proba(self, X):
        """
        Predict class probabilities.
        Note: ANFIS is inherently a regressor, so this is an approximation based on distance.
        """
        if not hasattr(self, "best_params_"):
            raise Exception("Perform a fit first.")

        X = check_array(X)
        y_pred_raw = self._forward_pass(X, self.antecedents_, self.consequents_)

        # Simple distance-based probability
        # This is a naive implementation; a softmax on rule outputs would be more 'correct'
        # but ANFIS structure doesn't output logits directly.
        # Here we just return the raw regression value as a 'score'

        # Alternatively, we can construct a dummy proba matrix
        # Distance to class 0, Distance to class 1...

        probas = np.zeros((len(X), len(self.classes_)))
        for i in range(len(self.classes_)):
            # 1 / (1 + distance)
            dist = np.abs(y_pred_raw - i)
            probas[:, i] = 1.0 / (1.0 + dist)

        return p_normalize(probas, axis=1)
