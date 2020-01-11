from __future__ import print_function

import numpy as np
import logging
import fylearn.local_search as ls

logging.basicConfig(level=logging.DEBUG)

def fitness(x):
    """
    A dummy fitness function. Find a solution with mean 15.
    """
    return np.abs(15.0 - np.mean(x))

def test_ps():

    upper_bound = np.array([20.0] * 10)
    lower_bound = np.array([10.0] * 10)

    o = ls.PatternSearchOptimizer(fitness, lower_bound, upper_bound, max_evaluations=50, random_state=1)

    best_sol, best_fit = o()

    print("best", best_sol)
    print("fitness", best_fit)

    assert best_fit < 0.05
    
def test_lus():

    upper_bound = np.array([20.0] * 10)
    lower_bound = np.array([10.0] * 10)

    o = ls.LocalUnimodalSamplingOptimizer(fitness, lower_bound, upper_bound,
                                          max_evaluations=50, gamma=3.0, random_state=1)

    best_sol, best_fit = o()

    print("best", best_sol)
    print("fitness", best_fit)

    assert best_fit < 0.1  # it's heuristic, so no guarantees here, we put the threshold a little high.

def test_helper_num_runs():

    upper_bound = np.array([20.0] * 10)
    lower_bound = np.array([10.0] * 10)

    o = ls.LocalUnimodalSamplingOptimizer(fitness, lower_bound, upper_bound,
                                          max_evaluations=25, gamma=3.0, random_state=1)

    best_sol, best_fit = o()

    print("best", best_sol)
    print("fitness", best_fit)

    new_sol, new_fit = ls.helper_num_runs(o, num_runs=100)

    # assume we can find better solution in 100 tries.
    assert new_fit < best_fit

    #
    # try pattern search with refine method
    #

    o = ls.PatternSearchOptimizer(fitness, lower_bound, upper_bound, max_evaluations=25, random_state=1)
    best_sol, best_fit = o()

    print("best", best_sol)
    print("fitness", best_fit)

    new_sol, new_fit = ls.helper_num_runs(o, num_runs=100, refine=ls.scipy_refine)

    assert new_fit < best_fit
