# -*- coding: utf-8 -*-
"""Fuzzy pattern classifier with genetic algorithm based methods

The module structure is the following:

- The "FuzzyPatternClassifierGA" is a FPC where the membership
  functions are learned using genetic algorithms.

References:

[1] Davidsen, 2014.
  
"""

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_arrays, column_or_1d
from sklearn.metrics import accuracy_score
import fylearn.fuzzylogic as fl
from fylearn.ga import GeneticAlgorithm

#
# Authors: Søren Atmakuri Davidsen <sorend@gmail.com>
#

# default aggregation rules to use
AGGREGATION_RULES = (fl.prod, np.nanmin, fl.mean, np.nanmax, fl.algebraic_sum)

# requires 1 gene
def build_aggregation(rules, chromosome, idx):
    i = int(chromosome[idx] * len(rules))
    if i < 0: i = 0
    if i >= len(rules): i = len(rules) - 1
    return rules[i]

# requires 3 genes
def build_pi_membership(chromosome, idx):
    a, r, b = sorted(chromosome[idx:idx+3])
    return fl.PiSet(a, r, b)

# requires 4 genes
def build_trapezoidal_membership(chromosome, idx):
    a, b, c, d = sorted(chromosome[idx:idx+4])
    return fl.TrapezoidalSet(a, b, c, d)

def build_t_membership(chromosome, idx):
    a, b, c = sorted(chromosome[idx:idx+3])
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
MEMBERSHIP_FACTORIES = (build_pi_membership, build_trapezoidal_membership, build_static_membership)

# requires 1 gene
def build_membership(mu_factories, chromosome, idx):
    i = int(chromosome[idx] * len(mu_factories))
    if i < 0: i = 0
    if i >= len(mu_factories): i = len(mu_factories) - 1
    return mu_factories[i](chromosome, idx+1)

# decodes aggregation and protos from chromosome
def _decode(m, aggregation_rules, mu_factories, classes, chromosome):
    aggregation = build_aggregation(aggregation_rules, chromosome, 0)
    protos = {}
    idx = 2
    for i in range(len(classes)):
        protos[i] = [ build_membership(mu_factories, chromosome, 2 + (i * m * 5) + (j * 4)) for j in range(m) ]
    return aggregation, protos

def _predict(prototypes, aggregation, classes, X):
    Mus = np.zeros(X.shape)
    R = np.zeros((X.shape[0], len(classes))) # holds output for each class
    attribute_idxs = range(X.shape[1])

    # class_idx has class_prototypes membership functions
    for class_idx, class_prototypes in prototypes.items():
        for i in attribute_idxs:
            Mus[:,i] = class_prototypes[i](X[:,i])
        R[:,class_idx] = aggregation(Mus)

    return classes.take(np.argmax(R, 1))

logger = logging.getLogger("fpcga")

class FuzzyPatternClassifierGA(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {"iterations": self.iterations,
                "epsilon": self.epsilon,
                "mu_factories": self.mu_factories,
                "aggregation_rules": self.aggregation_rules}

    def set_params(self, **kwargs):
        for key, value in params.items():
            self.setattr(key, value)
        return self
    
    def __init__(self, mu_factories=MEMBERSHIP_FACTORIES, aggregation_rules=AGGREGATION_RULES,
                 iterations=10, epsilon=0.0001):
        self.mu_factories = mu_factories
        self.aggregation_rules = aggregation_rules
        self.iterations = iterations
        self.epsilon = epsilon

    def fit(self, X, y_orig):

        X, = check_arrays(X)

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
        
        X, = check_arrays(X)

        # predict
        return _predict(self.protos_, self.aggregation, self.classes_, X)

    def build_with_ga(self, X, y):

        # root mean squared error fitness function
        def rmse_fitness_function(chromosome):
            # decode the class model from gene
            aggregation, mus = _decode(self.m, self.aggregation_rules, self.mu_factories, self.classes_, chromosome)

            y_pred = _predict(mus, aggregation, self.classes_, X)
            return 1.0 - accuracy_score(y, y_pred)

        # number of genes (2 for the aggregation, 4 for each attribute)
        n_genes = 2 + (self.m * 5 * len(self.classes_))

        logger.info("initializing GA %d iterations" % (self.iterations,))
        # initialize
        ga = GeneticAlgorithm(fitness_function=rmse_fitness_function,
                              scaling=1.0,
                              crossover_points=range(2, n_genes, 5),
                              keep_parents=10,
                              n_chromosomes=100,
                              n_genes=n_genes,
                              p_mutation=0.3)

        #print "population", ga.population_
        #print "fitness", ga.fitness_

        chromosomes, fitnesses = ga.best(10)
        last_fitness = np.mean(fitnesses)
        
        #
        for generation in range(self.iterations):
            ga.next()
            logger.info("GA iteration %d Fitness (top-4) %s" % (generation, str(ga.fitness_[:4])))
            chromosomes, fitnesses = ga.best(10)
            aggregation, protos = _decode(self.m, self.aggregation_rules, self.mu_factories, self.classes_, chromosomes[0])
            self.aggregation = aggregation
            self.protos_ = protos

            # check stopping condition
            new_fitness = np.mean(fitnesses)
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


if __name__ == "__main__":

    import sys
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cross_validation import cross_val_score
    
    data = pd.read_csv(sys.argv[1])
    X = data.ix[:,:-1]
    y = data["class"]

    # scale to [0, 1]
    X = MinMaxScaler().fit_transform(X)

    l = FuzzyPatternClassifierGA(iterations=100)

    # cross validation    
    scores = cross_val_score(l, X, y, cv=10)

    print "- dataset", sys.argv[1]
    print "- learner", l
    print "- scores %s, mean %f, std %f" % \
        (str(scores), np.mean(scores), np.std(scores))

    
