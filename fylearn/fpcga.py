# -*- coding: utf-8 -*-
"""Fuzzy pattern classifier with genetic algorithm based methods

The module structure is the following:

- The "FuzzyPatternClassifierGA" is a FPC where the membership
  functions are learned using genetic algorithms in global scheme [1]
- The "FuzzyPatternClassifierLGA" also learns mus using a GA, but in
  local scheme [1].

References:
[1] Davidsen, 2015.
"""

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.metrics import accuracy_score, mean_squared_error
import fylearn.fuzzylogic as fl
from fylearn.ga import GeneticAlgorithm, helper_fitness, UniformCrossover, helper_n_generations
from fylearn.ga import UnitIntervalGeneticAlgorithm

#
# Authors: Søren Atmakuri Davidsen <sorend@gmail.com>
#

# default aggregation rules to use
AGGREGATION_RULES = (fl.prod, fl.mean)

# requires 1 gene
def build_aggregation(X, y, rules, chromosome, idx):
    i = int(chromosome[idx] * len(rules))
    if i < 0:
        i = 0
    if i >= len(rules):
        i = len(rules) - 1
    return rules[i](X, y)

# requires 3 genes
def build_pi_membership(chromosome, idx):
    a, r, b = sorted(chromosome[idx:idx + 3])
    return fl.PiSet(a=a, r=r, b=b)

# requires 4 genes
def build_trapezoidal_membership(chromosome, idx):
    a, b, c, d = sorted(chromosome[idx:idx + 4])
    return fl.TrapezoidalSet(a, b, c, d)

def build_t_membership(chromosome, idx):
    a, b, c = sorted(chromosome[idx:idx + 3])
    return fl.TriangularSet(a, b, c)

class StaticFunction:
    def __call__(self, X):
        return 0.5

    def __str__(self):
        return "S(0.5)"

# requires 0 genes
def build_static_membership(chromosome, idx):
    return StaticFunction()

# default definition of membership function factories
MEMBERSHIP_FACTORIES = (build_pi_membership,)

# requires 1 gene
def build_membership(mu_factories, chromosome, idx):
    i = int(chromosome[idx] * len(mu_factories))
    if i < 0:
        i = 0
    if i >= len(mu_factories):
        i = len(mu_factories) - 1
    return mu_factories[i](chromosome, idx + 1)

# decodes aggregation and protos from chromosome
def _decode(m, X, y, aggregation_rules, mu_factories, classes, chromosome):
    aggregation = build_aggregation(X, y, aggregation_rules, chromosome, 0)
    protos = {}
    for i in range(len(classes)):
        protos[i] = [ build_membership(mu_factories, chromosome, 2 + (i * m * 5) + (j * 4)) for j in range(m) ]
    return aggregation, protos

def _predict_one(prototype, aggregation, X):
    Mus = np.zeros(X.shape)
    for i in range(X.shape[1]):
        Mus[:, i] = prototype[i](X[:, i])
    return aggregation(Mus)

def _predict(prototypes, aggregation, classes, X):
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
    pass

class DummyAggregationRuleFactory(AggregationRuleFactory):
    def __init__(self, aggregation_rule):
        self.aggregation_rule = aggregation_rule

    def __call__(self, X, y):
        return self.aggregation_rule

class FuzzyPatternClassifierGA(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {"iterations": self.iterations,
                "epsilon": self.epsilon,
                "mu_factories": self.mu_factories,
                "aggregation_rules": self.aggregation_rules}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.setattr(key, value)
        return self

    def __init__(self, mu_factories=MEMBERSHIP_FACTORIES, aggregation_rules=AGGREGATION_RULES,
                 iterations=10, epsilon=0.0001):

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

    def fit(self, X, y_orig):

        def as_factory(r):
            return r if isinstance(r, AggregationRuleFactory) else DummyAggregationRuleFactory(r)

        self.aggregation_rules__ = [ as_factory(r) for r in self.aggregation_rules ]
        
        X = check_array(X)

        self.classes_, y = np.unique(y_orig, return_inverse=True)
        self.m = X.shape[1]

        if np.nan in self.classes_:
            raise "nan not supported for class values"

        self.build_with_ga(X, y_orig)

        return self

    def predict(self, X):
        """

        Predict outputs given examples.

        Parameters:
        -----------

        X : the examples to predict (array or matrix)

        Returns:
        --------

        y_pred : Predicted values for each row in matrix.

        """
        if self.protos_ is None:
            raise Exception("Prototypes not initialized. Perform a fit first.")

        X = check_array(X)

        # predict
        return _predict(self.protos_, self.aggregation, self.classes_, X)

    def build_with_ga(self, X, y):

        # accuracy fitness function
        def accuracy_fitness_function(chromosome):
            # decode the class model from gene
            aggregation, mus = _decode(self.m, X, y, self.aggregation_rules__,
                                       self.mu_factories, self.classes_, chromosome)
            y_pred = _predict(mus, aggregation, self.classes_, X)
            return 1.0 - accuracy_score(y, y_pred)

        # number of genes (2 for the aggregation, 4 for each attribute)
        n_genes = 2 + (self.m * 5 * len(self.classes_))

        logger.info("initializing GA %d iterations" % (self.iterations,))
        # initialize
        ga = GeneticAlgorithm(fitness_function=helper_fitness(accuracy_fitness_function),
                              scaling=1.0,
                              crossover_function=UniformCrossover(0.5),
                              # crossover_points=range(2, n_genes, 5),
                              elitism=5,  # no elitism
                              n_chromosomes=100,
                              n_genes=n_genes,
                              p_mutation=0.3)

        last_fitness = None

        #
        for generation in range(self.iterations):
            ga.next()
            logger.info("GA iteration %d Fitness (top-4) %s" % (generation, str(np.sort(ga.fitness_)[:4])))
            chromosomes, fitnesses = ga.best(10)
            aggregation, protos = _decode(self.m, X, y, self.aggregation_rules__,
                                          self.mu_factories, self.classes_, chromosomes[0])
            self.aggregation = aggregation
            self.protos_ = protos

            # check stopping condition
            new_fitness = np.mean(fitnesses)
            if last_fitness is not None:
                d_fitness = last_fitness - new_fitness
                if self.epsilon is not None and d_fitness < self.epsilon:
                    logger.info("Early stop d_fitness %f" % (d_fitness,))
                    break
            last_fitness = new_fitness

        # print learned.
        logger.info("+- Final: Aggregation %s" % (str(self.aggregation),))
        for key, value in self.protos_.items():
            logger.info("`- Class-%d" % (key,))
            logger.info("`- Membership-fs %s" % (str([ x.__str__() for x in value ]),))

    def __str__(self):
        if self.protos_ is None:
            return "Not trained"
        else:
            return str(self.aggregation) + str({ "class-" + str(k): v for k, v in self.protos_ })

class FuzzyPatternClassifierLGA(FuzzyPatternClassifierGA):

    def decode(self, chromosome):
        return [ build_membership(self.mu_factories, chromosome, i * 5) for i in range(self.m) ]

    def build_for_class(self, X, y, class_idx):

        y_target = np.zeros(y.shape)  # create the target of 1 and 0.
        y_target[class_idx] = 1.0

        n_genes = 5 * self.m

        def rmse_fitness_function(chromosome):
            proto = self.decode(chromosome)
            y_pred = _predict_one(proto, self.aggregation, X)
            return mean_squared_error(y_target, y_pred)

        logger.info("initializing GA %d iterations" % (self.iterations,))
        # initialize
        ga = GeneticAlgorithm(fitness_function=helper_fitness(rmse_fitness_function),
                              scaling=1.0,
                              crossover_function=UniformCrossover(0.5),
                              # crossover_points=range(0, n_genes, 5),
                              elitism=5,  # no elitism
                              n_chromosomes=100,
                              n_genes=n_genes,
                              p_mutation=0.3)

        # print "population", ga.population_
        # print "fitness", ga.fitness_

        chromosomes, fitnesses = ga.best(10)
        last_fitness = np.mean(fitnesses)

        proto = None
        #
        for generation in range(self.iterations):
            ga.next()
            logger.info("GA iteration %d Fitness (top-4) %s" % (generation, str(ga.fitness_[:4])))
            chromosomes, fitnesses = ga.best(10)
            proto = self.decode(chromosomes[0])

            # check stopping condition
            new_fitness = np.mean(fitnesses)
            d_fitness = last_fitness - new_fitness
            if self.epsilon is not None and d_fitness < self.epsilon:
                logger.info("Early stop d_fitness %f" % (d_fitness,))
                break
            last_fitness = new_fitness

        return proto

    def build_with_ga(self, X, y):
        self.aggregation = self.aggregation_rules__[0](X, y)
        self.protos_ = {}
        for class_no, class_value in enumerate(self.classes_):
            class_idx = np.array(y == class_value)

            proto = self.build_for_class(X, y, class_idx)
            self.protos_[class_no] = proto

        # print learned.
        logger.info("+- Final: Aggregation %s" % (str(self.aggregation),))
        for key, value in self.protos_.items():
            logger.info("`- Class-%d" % (key,))
            logger.info("`- Membership-fs %s" % (str([ x.__str__() for x in value ]),))


class SEFuzzyPatternClassifier(FuzzyPatternClassifierGA):

    def get_params(self, deep=False):
        return {"iterations": self.iterations,
                "aggregation": self.aggregation,
                "adjust_center": self.adjust_center}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.setattr(key, value)
        return self

    def __init__(self, aggregation=fl.prod, iterations=25, adjust_center=False):
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

        assert iterations > 0

    def build_for_class(self, X, y, class_idx):

        # take column-wise min/mean/max for class
        mins = np.nanmin(X[class_idx], 0)
        means = np.nanmean(X[class_idx], 0)
        maxs = np.nanmax(X[class_idx], 0)
        ds = (maxs - mins) / 2.0

        n_genes = 2 * self.m  # adjustment for r and shrinking/expanding value for p/q

        B = np.ones(n_genes)

        def decode_with_shrinking_expanding(C):
            def dcenter(j):
                return min(1.0, max(0.0, C[j])) - 0.5 if self.adjust_center else 1.0

            return [ fl.PiSet(r=means[j] * dcenter(j),
                              p=means[j] - (ds[j] * C[j + 1]),
                              q=means[j] + (ds[j] * C[j + 1]))
                     for j in range(self.m) ]

        y_target = np.zeros(y.shape)  # create the target of 1 and 0.
        y_target[class_idx] = 1.0

        def rmse_fitness_function(chromosome):
            proto = decode_with_shrinking_expanding(chromosome)
            y_pred = _predict_one(proto, self.aggregation, X)
            return mean_squared_error(y_target, y_pred)

        logger.info("initializing GA %d iterations" % (self.iterations,))
        # initialize
        ga = UnitIntervalGeneticAlgorithm(fitness_function=helper_fitness(rmse_fitness_function),
                                          crossover_function=UniformCrossover(0.5),
                                          elitism=3,
                                          n_chromosomes=100,
                                          n_genes=n_genes,
                                          p_mutation=0.3)

        ga = helper_n_generations(ga, self.iterations)
        chromosomes, fitnesses = ga.best(1)

        return decode_with_shrinking_expanding(chromosomes[0]), decode_with_shrinking_expanding(B)

    def build_with_ga(self, X, y):
        self.protos_ = {}
        self.bases_ = {}
        for class_no, class_value in enumerate(self.classes_):
            class_idx = np.array(y == class_value)

            proto, base = self.build_for_class(X, y, class_idx)
            self.protos_[class_no] = proto
            self.bases_[class_no] = base

    def toggle_base(self):
        if hasattr(self, "backups_"):
            self.protos_ = self.backups_
            del self.backups_
        else:
            self.backups_ = self.protos_
            self.protos_ = self.bases_
        return self
