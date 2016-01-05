
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

    o = ls.PatternSearchOptimizer(fitness, lower_bound, upper_bound, max_evaluations=50,
                                  num_runs=100, refine_function=ls.scipy_refine)

    best_sol, best_fit = o()

    print "best", best_sol
    print "fitness", best_fit

    assert best_fit < 0.01
    
def test_lus():

    upper_bound = np.array([20.0] * 10)
    lower_bound = np.array([10.0] * 10)

    o = ls.LocalUnimodalSamplingOptimizer(fitness, lower_bound, upper_bound,
                                          max_evaluations=50, num_runs=100,
                                          gamma=3.0)

    best_sol, best_fit = o()

    print "best", best_sol
    print "fitness", best_fit

    assert best_fit < 0.01
