"""Fuzzy pattern classifier with genetic algorithm based methods

The module structure is the following:

- The "FuzzyPatternClassifierGA" is a FPC where the membership
  functions are learned using genetic algorithms in global scheme [1]
- The "FuzzyPatternClassifierLGA" also learns mus using a GA, but in
  local scheme [1].

References:
-----------
[1] S. A. Davidsen, E. Sreedevi, and M. Padmavathamma, "Local and global genetic fuzzy pattern
    classifiers," In Proc. Machine Learning and Data Mining in Pattern Recognition, pp. 55-69,
    2015. url: https://link.springer.com/chapter/10.1007/978-3-319-21024-7_4

"""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.validation import check_array

import fylearn.fuzzylogic as fl
from fylearn._validation import has_nan_classes
from fylearn.ga import (
    GeneticAlgorithm,
    UniformCrossover,
    UnitIntervalGeneticAlgorithm,
    helper_fitness,
    helper_n_generations,
)

#
# Authors: Søren Atmakuri Davidsen <sorend@gmail.com>
#

# default aggregation rules to use
AGGREGATION_RULES = (fl.prod, fl.mean)


# requires 1 gene
def build_aggregation(X: np.ndarray, y: np.ndarray, rules: Sequence[Any], chromosome: np.ndarray, idx: int) -> Any:
    i = int(chromosome[idx] * len(rules))
    if i < 0:
        i = 0
    if i >= len(rules):
        i = len(rules) - 1
    return rules[i](X, y)


# requires 3 genes
def build_pi_membership(chromosome: np.ndarray, idx: int) -> fl.PiSet:
    a, r, b = sorted(chromosome[idx : idx + 3])
    return fl.PiSet(a=a, r=r, b=b)


# requires 4 genes
def build_trapezoidal_membership(chromosome: np.ndarray, idx: int) -> fl.TrapezoidalSet:
    a, b, c, d = sorted(chromosome[idx : idx + 4])
    return fl.TrapezoidalSet(a, b, c, d)


def build_t_membership(chromosome: np.ndarray, idx: int) -> fl.TriangularSet:
    a, b, c = sorted(chromosome[idx : idx + 3])
    return fl.TriangularSet(a, b, c)


class StaticFunction:
    def __call__(self, X: Any) -> float:
        return 0.5

    def __str__(self) -> str:
        return "S(0.5)"


# requires 0 genes
def build_static_membership(chromosome: np.ndarray, idx: int) -> StaticFunction:
    return StaticFunction()


# default definition of membership function factories
MEMBERSHIP_FACTORIES = (build_pi_membership,)


# Global FPCGA chromosome schema: one aggregation selector followed by one
# four-gene block (factory selector + three parameters) per class and feature.
AGGREGATION_GENE_COUNT = 1
MEMBERSHIP_GENE_STRIDE = 4


def chromosome_size(m: int, n_classes: int) -> int:
    return AGGREGATION_GENE_COUNT + m * n_classes * MEMBERSHIP_GENE_STRIDE


def membership_gene_index(class_idx: int, feature_idx: int, m: int) -> int:
    return AGGREGATION_GENE_COUNT + (class_idx * m + feature_idx) * MEMBERSHIP_GENE_STRIDE


# requires 1 gene
def build_membership(
    mu_factories: Sequence[Callable], chromosome: np.ndarray, idx: int
) -> Any:
    i = int(chromosome[idx] * len(mu_factories))
    if i < 0:
        i = 0
    if i >= len(mu_factories):
        i = len(mu_factories) - 1
    return mu_factories[i](chromosome, idx + 1)


# decodes aggregation and protos from chromosome
def _decode(
    m: int,
    X: np.ndarray,
    y: np.ndarray,
    aggregation_rules: Sequence[Any],
    mu_factories: Sequence[Callable],
    classes: np.ndarray,
    chromosome: np.ndarray,
) -> tuple[Any, dict[int, list[Any]]]:
    expected = chromosome_size(m, len(classes))
    if len(chromosome) != expected:
        raise ValueError(f"expected chromosome with {expected} genes, got {len(chromosome)}")
    aggregation = build_aggregation(X, y, aggregation_rules, chromosome, 0)
    protos = {}
    for i in range(len(classes)):
        protos[i] = [
            build_membership(mu_factories, chromosome, membership_gene_index(i, j, m)) for j in range(m)
        ]
    return aggregation, protos


def _predict_one(prototype: list[Any], aggregation: Callable[..., Any], X: np.ndarray) -> np.ndarray:
    Mus = np.zeros(X.shape)
    for i in range(X.shape[1]):
        Mus[:, i] = prototype[i](X[:, i])
    return aggregation(Mus)


def _predict(
    prototypes: dict[int, list[Any]], aggregation: Callable[..., Any], classes: np.ndarray, X: np.ndarray
) -> np.ndarray:
    Mus = np.zeros(X.shape)
    R = np.zeros((X.shape[0], len(classes)))  # holds output for each class
    attribute_idxs = range(X.shape[1])

    # class_idx has class_prototypes membership functions
    for class_idx, class_prototypes in prototypes.items():
        for i in attribute_idxs:
            Mus[:, i] = class_prototypes[i](X[:, i])
        R[:, class_idx] = aggregation(Mus)

    return classes.take(np.argmax(R, 1))


logger = logging.getLogger("fpcga")


class AggregationRuleFactory:
    def __call__(self, X: np.ndarray, y: np.ndarray) -> Any:
        raise NotImplementedError("subclasses must implement __call__")


class DummyAggregationRuleFactory(AggregationRuleFactory):
    def __init__(self, aggregation_rule):
        self.aggregation_rule = aggregation_rule

    def __call__(self, X, y):
        return self.aggregation_rule


class FuzzyPatternClassifierGA(BaseEstimator, ClassifierMixin):
    protos_: dict[int, list[Any]]
    aggregation: Any
    aggregation_rules__: Sequence[Any]

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "epsilon": self.epsilon,
            "mu_factories": self.mu_factories,
            "aggregation_rules": self.aggregation_rules,
            "random_state": self.random_state,
        }

    def set_params(self, **kwargs: Any) -> "FuzzyPatternClassifierGA":
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def __init__(
        self,
        mu_factories: Sequence[Callable] = MEMBERSHIP_FACTORIES,
        aggregation_rules: Sequence[Any] = AGGREGATION_RULES,
        iterations: int = 10,
        epsilon: float | None = 0.0001,
        random_state: Any = None,
    ):
        if mu_factories is None or len(mu_factories) == 0:
            raise ValueError("no mu_factories specified")

        if aggregation_rules is None or len(aggregation_rules) == 0:
            raise ValueError("no aggregation_rules specified")

        if iterations <= 0:
            raise ValueError("iterations must be > 0")

        self.mu_factories = mu_factories
        self.iterations = iterations
        self.epsilon = epsilon
        self.aggregation_rules = aggregation_rules
        self.random_state = random_state

    def fit(self, X: Any, y_orig: Any) -> "FuzzyPatternClassifierGA":
        def as_factory(r: Any) -> Any:
            return r if isinstance(r, AggregationRuleFactory) else DummyAggregationRuleFactory(r)

        self.aggregation_rules__ = [as_factory(r) for r in self.aggregation_rules]

        X = check_array(X)

        self.classes_, _ = np.unique(y_orig, return_inverse=True)
        self.m = X.shape[1]

        if has_nan_classes(self.classes_):
            raise Exception("nan not supported for class values")

        self.build_with_ga(X, y_orig)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """

        Predict outputs given examples.

        Parameters:
        -----------

        X : the examples to predict (array or matrix)

        Returns:
        --------

        y_pred : Predicted values for each row in matrix.

        """
        if not hasattr(self, "protos_"):
            raise Exception("Prototypes not initialized. Perform a fit first.")

        X = check_array(X)

        # predict
        return _predict(self.protos_, self.aggregation, self.classes_, X)

    def build_with_ga(self, X: np.ndarray, y: np.ndarray) -> None:
        # accuracy fitness function
        def accuracy_fitness_function(chromosome: np.ndarray) -> float:
            # decode the class model from gene
            aggregation, mus = _decode(
                self.m, X, y, self.aggregation_rules__, self.mu_factories, self.classes_, chromosome
            )
            y_pred = _predict(mus, aggregation, self.classes_, X)
            return 1.0 - accuracy_score(y, y_pred)

        n_genes = chromosome_size(self.m, len(self.classes_))

        logger.info(f"initializing GA {self.iterations} iterations")
        # initialize
        ga = GeneticAlgorithm(
            fitness_function=helper_fitness(accuracy_fitness_function),
            scaling=1.0,
            crossover_function=UniformCrossover(0.5),
            # crossover_points=range(1, n_genes, MEMBERSHIP_GENE_STRIDE),
            elitism=5,  # no elitism
            n_chromosomes=100,
            n_genes=n_genes,
            p_mutation=0.3,
            random_state=self.random_state,
        )

        last_fitness = None

        #
        for generation in range(self.iterations):
            next(ga)
            logger.info(f"GA iteration {generation} Fitness (top-4) {str(np.sort(ga.fitness_)[:4])}")
            chromosomes, fitnesses = ga.best(10)
            aggregation, protos = _decode(
                self.m, X, y, self.aggregation_rules__, self.mu_factories, self.classes_, chromosomes[0]
            )
            self.aggregation = aggregation
            self.protos_ = protos

            # check stopping condition
            new_fitness = np.mean(fitnesses)
            if last_fitness is not None:
                d_fitness = last_fitness - new_fitness
                if self.epsilon is not None and d_fitness < self.epsilon:
                    logger.info(f"Early stop d_fitness {d_fitness:f}")
                    break
            last_fitness = new_fitness

        # print learned.
        logger.info(f"+- Final: Aggregation {str(self.aggregation)}")
        for key, value in self.protos_.items():
            logger.info(f"`- Class-{key}")
            logger.info(f"`- Membership-fs {str([x.__str__() for x in value])}")

    def __str__(self):
        if not hasattr(self, "protos_"):
            return "Not trained"
        else:
            return str(self.aggregation) + str({"class-" + str(k): v for k, v in self.protos_.items()})


class FuzzyPatternClassifierLGA(FuzzyPatternClassifierGA):
    def decode(self, chromosome: np.ndarray) -> list[Any]:
        return [build_membership(self.mu_factories, chromosome, i * MEMBERSHIP_GENE_STRIDE) for i in range(self.m)]

    def build_for_class(self, X: np.ndarray, y: np.ndarray, class_idx: np.ndarray) -> list[Any]:
        y_target = np.zeros(y.shape)  # create the target of 1 and 0.
        y_target[class_idx] = 1.0

        n_genes = MEMBERSHIP_GENE_STRIDE * self.m

        def rmse_fitness_function(chromosome: np.ndarray) -> float:
            proto = self.decode(chromosome)
            y_pred = _predict_one(proto, self.aggregation, X)
            return mean_squared_error(y_target, y_pred)

        logger.info(f"initializing GA {self.iterations} iterations")
        # initialize
        ga = GeneticAlgorithm(
            fitness_function=helper_fitness(rmse_fitness_function),
            scaling=1.0,
            crossover_function=UniformCrossover(0.5),
            # crossover_points=range(0, n_genes, MEMBERSHIP_GENE_STRIDE),
            elitism=5,  # no elitism
            n_chromosomes=100,
            n_genes=n_genes,
            p_mutation=0.3,
        )

        # print "population", ga.population_
        # print "fitness", ga.fitness_

        chromosomes, fitnesses = ga.best(10)
        last_fitness = np.mean(fitnesses)

        proto: list[Any] = []
        #
        for generation in range(self.iterations):
            next(ga)
            logger.info(f"GA iteration {generation} Fitness (top-4) {str(ga.fitness_[:4])}")
            chromosomes, fitnesses = ga.best(10)
            proto = self.decode(chromosomes[0])

            # check stopping condition
            new_fitness = np.mean(fitnesses)
            d_fitness = last_fitness - new_fitness
            if self.epsilon is not None and d_fitness < self.epsilon:
                logger.info(f"Early stop d_fitness {d_fitness:f}")
                break
            last_fitness = new_fitness

        return proto

    def build_with_ga(self, X: np.ndarray, y: np.ndarray) -> None:
        self.aggregation = self.aggregation_rules__[0](X, y)
        self.protos_ = {}
        for class_no, class_value in enumerate(self.classes_):
            class_idx = np.array(y == class_value)

            proto = self.build_for_class(X, y, class_idx)
            self.protos_[class_no] = proto

        # print learned.
        logger.info(f"+- Final: Aggregation {str(self.aggregation)}")
        for key, value in self.protos_.items():
            logger.info(f"`- Class-{key}")
            logger.info(f"`- Membership-fs {str([x.__str__() for x in value])}")


class SEFuzzyPatternClassifier(FuzzyPatternClassifierGA):
    bases_: dict[int, list[Any]]
    backups_: dict[int, list[Any]]

    def get_params(self, deep=False):
        return {"iterations": self.iterations, "aggregation": self.aggregation, "adjust_center": self.adjust_center}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def __init__(self, aggregation: Callable[..., Any] = fl.prod, iterations: int = 25, adjust_center: bool = False):
        """
        Constructs classifier

        Parameters:
        -----------

        aggregation : fuzzy aggregation to use.

        iterations : number of iterations for the GA.

        adjust_center : Allow to adjust center of the membership function.
        """
        self.aggregation = aggregation
        self.iterations = iterations
        self.adjust_center = adjust_center
        # used by the inherited fit() implementation
        self.aggregation_rules = (self.aggregation,)

        assert iterations > 0

    def build_for_class(self, X: np.ndarray, y: np.ndarray, class_idx: np.ndarray) -> tuple[list[Any], list[Any]]:
        # take column-wise min/mean/max for class
        mins = np.nanmin(X[class_idx], 0)
        means = np.nanmean(X[class_idx], 0)
        maxs = np.nanmax(X[class_idx], 0)
        ds = (maxs - mins) / 2.0

        n_genes = 2 * self.m  # adjustment for r and shrinking/expanding value for p/q

        B = np.ones(n_genes)

        def decode_with_shrinking_expanding(C: np.ndarray) -> list[Any]:
            def dcenter(j: int) -> float:
                return min(1.0, max(0.0, C[j])) - 0.5 if self.adjust_center else 1.0

            return [
                fl.PiSet(r=means[j] * dcenter(j), p=means[j] - (ds[j] * C[j + 1]), q=means[j] + (ds[j] * C[j + 1]))
                for j in range(self.m)
            ]

        y_target = np.zeros(y.shape)  # create the target of 1 and 0.
        y_target[class_idx] = 1.0

        def rmse_fitness_function(chromosome: np.ndarray) -> float:
            proto = decode_with_shrinking_expanding(chromosome)
            y_pred = _predict_one(proto, self.aggregation, X)
            return mean_squared_error(y_target, y_pred)

        logger.info(f"initializing GA {self.iterations} iterations")
        # initialize
        ga = UnitIntervalGeneticAlgorithm(
            fitness_function=helper_fitness(rmse_fitness_function),
            crossover_function=UniformCrossover(0.5),
            elitism=3,
            n_chromosomes=100,
            n_genes=n_genes,
            p_mutation=0.3,
        )

        ga = helper_n_generations(ga, self.iterations)
        chromosomes, fitnesses = ga.best(1)

        return decode_with_shrinking_expanding(chromosomes[0]), decode_with_shrinking_expanding(B)

    def build_with_ga(self, X: np.ndarray, y: np.ndarray) -> None:
        self.protos_ = {}
        self.bases_: dict[int, list[Any]] = {}
        for class_no, class_value in enumerate(self.classes_):
            class_idx = np.array(y == class_value)

            proto, base = self.build_for_class(X, y, class_idx)
            self.protos_[class_no] = proto
            self.bases_[class_no] = base

    def toggle_base(self) -> "SEFuzzyPatternClassifier":
        if hasattr(self, "backups_"):
            self.protos_ = self.backups_
            del self.backups_
        else:
            self.backups_ = self.protos_
            self.protos_ = self.bases_
        return self
