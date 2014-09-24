# -*- coding: utf-8 -*-
"""Fuzzy reduction rule based methods

The module structure is the following:

- The "FuzzyReductionRuleClassifier" implements the model learning using the
  [1, 2] algorithm.

References:

[1] Meher, 2009.
  
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_arrays, column_or_1d
from sklearn.metrics import mean_squared_error
import fylearn.fuzzylogic as fl
from fylearn.ga import GeneticAlgorithm
from fylearn.frr import pi_factory, FuzzyReductionRuleClassifier

#
# Authors: Søren Atmakuri Davidsen <sorend@gmail.com>
#

class FuzzyReductionRuleGAClassifier(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {"chromosomes": self.chromosomes,
                "membership_factory": self.membership_factory}

    def set_params(self, **kwargs):
        for key, value in params.items():
            self.setattr(key, value)
        return self
    
    def __init__(self, aggregation=np.mean, membership_factory=pi_factory(2.0)):
        self.aggregation = aggregation
        self.membership_factory = membership_factory

    def decode_rule(self, g):
        if g >= 0:
            return np.mean
        else:
            return fl.prod
    
    def decode(self, chromosome):
        # encode chromosome into the frr classifier
        self.frr.aggregation = self.decode_rule(chromosome[0])
        idx = 1
        for class_ in self.classes_:
            # array of pi functions
            new_protos = [ self.membership_factory(chromosome[idx], chromosome[idx+1], chromosome[idx+2])
                           for pi_func in self.frr.protos_[class_] ]
            self.frr.protos_[class_] = new_protos
            idx += 3

    def init_population(self, X, y):
        # create initial population
        population = np.zeros((100, 1 + (X.shape[1] * 3 * len(self.classes_))))

        for idx, class_ in enumerate(self.classes_):
            mins  = np.nanmin(X[y == idx], 0)
            means = np.nanmean(X[y == idx], 0)
            maxs  = np.nanmax(X[y == idx], 0)

            offset = (3 * idx) + 1
            offset_r = offset + (X.shape[1] * 3)
            population[:,offset:offset_r][:,0::3] = mins
            population[:,offset:offset_r][:,1::3] = means
            population[:,offset:offset_r][:,2::3] = maxs


        population += ((np.random.random(population.shape) - 0.5) * self.scaling)
        return population

    
    def fit(self, X, y_orig):

        X, = check_arrays(X)

        self.classes_, y = np.unique(y_orig, return_inverse=True)

        if np.nan in self.classes_:
            raise "nan not supported for class values"

        # create and (initial) fit of inner classifier
        self.frr = FuzzyReductionRuleClassifier(self.aggregation, self.membership_factory)
        self.frr.fit(X, y)

        def mse_fitness_function(chromosome):
            # make prediction
            self.decode(chromosome)
            y_pred = self.frr.predict(X)
            return mean_squared_error(y, y_pred)

        # calculate scaling
        self.scaling = 10.0

        # initialize population
        population = self.init_population(X, y)
        
        # 3 genes for each feature (a, b, c in pi function)
        ga = GeneticAlgorithm(fitness_function=mse_fitness_function,
                              population=population,
                              scaling=self.scaling,
                              p_mutation=0.1)

        print "population", ga.population_
        print "fitness", ga.fitness_
        
        #
        for generation in range(5):
            ga.next()
            print "GA iteration", generation, ga.fitness_[:4]

        self.chromosome = ga.best(1)[0]
        print "selected", self.chromosome
        self.frr = self.decode(self.chromosome)

        return self

    def predict(self, X):
        return self.frr.predict(X)

