# -*- coding: utf-8 -*-
"""
Implementation of Teaching-Learning Based Optimization [1].

[1] R. V. Rao et al. "Teaching–learning-based optimization: A novel method for constrained
    mechanical design optimization problems." Computer-Aided Design 43(3): 303-315, 2011.

"""
import numpy as np
from sklearn.utils import check_random_state


#
# Authors: Søren A. Davidsen <soren@atmakuridavidsen.com>
#

class TeachingLearningBasedOptimizer(object):
    """
    Teaching-Learning based optimizer (TLBO).

    TLBO is a population based optimizer which works in two phases:

    1. Teaching phase: Using the best individual as teacher, randomly adjust other individuals
       using difference between teacher and population mean.

    2. Learner phase: Randomly adjust pairs of individuals based on the best one with best fitness.
    """
    def __init__(self, f, lower_bound, upper_bound,
                 n_population=50, random_state=None):
        """
        Constructor

        Parameters:
        -----------

        f : function to minimize.

        lower_bound : Vector with lower bound of the search space.

        upper_bound : Vector with upper bound of the search space.

        n_population : Number of individuals in the population [Default: 50].

        random_state : Specific random state to use [Default: None]
        """
        self.f = f
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.pidx = list(range(n_population))
        self.m = lower_bound.shape[0]  # columns
        self.random_state = check_random_state(random_state)

        # init population and fitness
        self.population_ = self.random_state.rand(n_population, self.m) * (upper_bound - lower_bound) + lower_bound
        self.fitness_ = np.apply_along_axis(self.f, 1, self.population_)
        # init bestidx
        self.bestidx_ = np.argmin(self.fitness_)
        self.bestcosts_ = [ self.fitness_[self.bestidx_] ]

    def best(self, num=1):
        """
        Returns the num best solution and fitness at current epoch
        """
        bestidxs = np.argsort(self.fitness_)[:num]
        return self.population_[bestidxs], self.fitness_[bestidxs]

    def next(self):
        """
        One iteration of TLBO.
        """
        mean = np.nanmean(self.population_, axis=0)  # column mean.
        rs = self.random_state

        teacher = np.argmin(self.fitness_)  # select teacher

        # teaching phase
        for i in self.pidx:
            T_F = rs.randint(1, 3)  # teaching factor is randomly 1 or 2.
            r_i = rs.rand(self.m)  # multiplier also random.

            new_solution = self.population_[i] + (r_i * (teacher - (T_F * mean)))
            new_solution = np.minimum(np.maximum(new_solution, self.lower_bound), self.upper_bound)

            new_fitness = self.f(new_solution)

            if new_fitness < self.fitness_[i]:
                self.population_[i] = new_solution
                self.fitness_[i] = new_fitness

        # learning phase
        for i in self.pidx:

            j = rs.choice(self.pidx[:i] + self.pidx[(i + 1):], 1)  # pick another random i!=j
            r_i = rs.rand(self.m)

            if self.fitness_[i] < self.fitness_[j]:
                new_solution = self.population_[i] + r_i * (self.population_[i] - self.population_[j])
            else:
                new_solution = self.population_[i] + r_i * (self.population_[j] - self.population_[i])

            new_solution = np.minimum(np.maximum(new_solution, self.lower_bound), self.upper_bound)
            new_fitness = self.f(new_solution)

            if new_fitness < self.fitness_[i]:
                self.population_[i] = new_solution
                self.fitness_[i] = new_fitness

        self.bestidx_ = np.argmin(self.fitness_)  # update details
        self.bestcosts_.append(self.fitness_[self.bestidx_])

TLBO = TeachingLearningBasedOptimizer  # shortcut to save our fingers
