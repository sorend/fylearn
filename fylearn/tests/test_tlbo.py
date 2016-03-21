import numpy as np
from fylearn.tlbo import TLBO, TeachingLearningBasedOptimizer
from fylearn.ga import helper_n_generations

def test_tlbo_variance():

    lb = np.zeros(5)
    lb[::2] = 1.1

    ub = np.ones(5)
    ub[::2] = 2.2

    o = TLBO(f=lambda x: np.abs(np.mean(x) - 0.5), lower_bounds=lb, upper_bounds=ub)

    o = helper_n_generations(o, 20)

    print "costs history", o.bestcosts_
    solution, fitness = o.best()
    print "best fitness", fitness
    print "best solution", solution

    assert len(o.fitness_) == 50
    assert len(o.population_) == 50
    assert len(o.bestcosts_) == 21

def test_tlbo_sphere():
    """Example given in matlab code"""

    o = TeachingLearningBasedOptimizer(f=lambda x: np.sum(x**2),
                                       lower_bounds=np.ones(10) * -10.0,
                                       upper_bounds=np.ones(10) * 10.0,
                                       n_population=34)

    o = helper_n_generations(o, 50)

    solution, fitness = o.best()

    print "costs history", o.bestcosts_
    print "best fitness", fitness
    print "best solution", solution

    assert len(o.fitness_) == 34
    assert len(o.population_) == 34
