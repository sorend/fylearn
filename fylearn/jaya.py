# -*- coding: utf-8 -*-
"""
Implementation of Jaya Algorithm Optimization [1].

[1] R. V. Rao, "Jaya: A simple and new optimization algorithm for solving constrained
    and unconstrained optimization problems." Int J of Industrial Engineering
     Computations, 2016, 7(1):19-34.
"""
import numpy as np
from sklearn.utils import check_random_state

#
# Authors: Søren A. Davidsen <soren@atmakuridavidsen.com>
#

class JayaOptimizer(object):
    """
    Jaya Algorithm is a population based optimizer which optimizes a function based on
    best and worst solutions in the population:

    1. Move towards the best solution

    2. Avoid the worst solution

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
        self.pidx = range(n_population)
        self.m = lower_bound.shape[0]  # columns
        self.random_state = check_random_state(random_state)

        # init population and fitness
        self.population_ = self.random_state.rand(n_population, self.m) * (upper_bound - lower_bound) + lower_bound
        self.fitness_ = np.apply_along_axis(self.f, 1, self.population_)
        # init bestidx
        self.bestidx_ = np.argmin(self.fitness_)
        self.bestcosts_ = [ self.fitness_[self.bestidx_] ]

    def best(self):
        """
        Returns the best solution and fitness at current epoch
        """
        return self.population_[self.bestidx_], self.fitness_[self.bestidx_]

    def next(self):
        """
        One iteration of Jaya Algorithm
        """
        rs = self.random_state

        # find best and worst
        fitness_sorted = np.argsort(self.fitness_)
        best, worst = self.population_[fitness_sorted[0]], self.population_[fitness_sorted[-1]]

        # update using best and worst
        for i in self.pidx:
            r1_i, r2_i = rs.rand(self.m), rs.rand(self.m)  # random modification vectors

            # make new solution
            new_solution = (
                self.population_[i] +                            # old position
                (r1_i * (best - np.abs(self.population_[i]))) -  # move towards best solution
                (r2_i * (worst - np.abs(self.population_[i])))   # and avoid worst
            )

            # bound
            new_solution = np.minimum(self.upper_bound, np.maximum(self.lower_bound, new_solution))

            new_fitness = self.f(new_solution)

            if new_fitness < self.fitness_[i]:
                self.population_[i] = new_solution
                self.fitness_[i] = new_fitness

        self.bestidx_ = np.argmin(self.fitness_)  # update details
        self.bestcosts_.append(self.fitness_[self.bestidx_])
