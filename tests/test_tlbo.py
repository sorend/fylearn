from __future__ import print_function
import numpy as np
from fylearn.tlbo import TLBO, TeachingLearningBasedOptimizer
from fylearn.ga import helper_n_generations
import pytest

def test_tlbo_variance():

    lb = np.zeros(5)
    lb[::2] = 1.1

    ub = np.ones(5)
    ub[::2] = 2.2

    o = TLBO(f=lambda x: np.abs(np.mean(x) - 0.5), lower_bound=lb, upper_bound=ub)

    o = helper_n_generations(o, 20)

    print("costs history", o.bestcosts_)
    solution, fitness = o.best()
    print("best fitness", fitness[0])
    print("best solution", solution[0])

    assert len(o.fitness_) == 50
    assert len(o.population_) == 50
    assert len(o.bestcosts_) == 21

    assert len(solution) == 1
    assert len(fitness) == 1


def test_tlbo_sphere():
    """Example given in matlab code"""

    o = TeachingLearningBasedOptimizer(f=lambda x: np.sum(x**2),
                                       lower_bound=np.ones(10) * -10.0,
                                       upper_bound=np.ones(10) * 10.0,
                                       n_population=34)

    o = helper_n_generations(o, 50)

    solution, fitness = o.best(3)

    print("costs history", o.bestcosts_)
    print("best fitness", fitness[0])
    print("best solution", solution[0])

    assert len(o.fitness_) == 34
    assert len(o.population_) == 34

    assert len(solution) == 3
    assert len(fitness) == 3

def test_tlbo_random_state():

    params = {'f':lambda x: np.sum(x**2),
             'lower_bound':np.ones(10) * -10.0,
             'upper_bound':np.ones(10) * 10.0,
             'n_population':34,
             'random_state':'wrong_value'
    }

    with pytest.raises(ValueError):
        TLBO(**params)

def test_tlbo_alan_2():

    def fitness(X):
        print("X", X, "len", len(X))
        x1, x2, x3, x4 = X.tolist()
        return 0.0358 +(0.7349*x1) + (.0578*x2)-(0.3151*x3) + (0.6888*x4) + (0.1803*(x1**2)) -(0.0481*(x2**2)) + (0.1699*(x3**2)) - (0.0494*(x4**2))- (0.3555*(x1*x2)) - (0.6316*(x1*x3)) + (0.5973*(x1*x4)) + (0.0826*(x2*x3)) - (0.4736*(x2*x4)) -  (0.6547*(x3*x4))    


    lower_bounds = np.array([-10,-10,-10,-10])
    upper_bounds = np.array([10,10,10,10])

    tlbo = TeachingLearningBasedOptimizer(fitness, lower_bounds, upper_bounds)
    tlbo = helper_n_generations(tlbo, 100)
    best_solution, best_fitness = tlbo.best()
    print("TLBO solution", best_solution, "fitness", best_fitness)

#
# Taken from paper:
# https://drive.google.com/file/d/0B96X2BLz4rx-OVpoaE5ucFEyanM/view
# See section 3.2 (Constrained optimization problems)
#
def test_tlbo_constrained_himmelblau():
    # function to minimize
    def f(x):
        x1, x2 = x
        return ((x1**2) + x2 - 11)**2 + (x1 + (x2**2) - 7)**2

    # constraint 2
    def g1(x):
        x1, x2 = x
        return 26 - (x1 - 5)**2 - x2**2

    # constraint 1 g2(x) >= 0
    def g2(x):
        x1, x2 = x
        return 20 - (4 * x1) - x2

    # penalty:
    #   10 * v^2, if v < 0
    #   0, otherwise
    def p(v):
        return 10 * min(v, 0)**2
    
    # f'(x) - add penalties for constraints
    def fp(x):
        return f(x) + p(g1(x)) + p(g2(x))

    lower_bounds = np.ones(2) * -5
    upper_bounds = np.ones(2) * 5

    tlbo = TLBO(fp, lower_bounds, upper_bounds, n_population=5)
    tlbo = helper_n_generations(tlbo, 100)
    best_solution, best_fitness = tlbo.best()
    print("TLBO solution", best_solution, "fitness", best_fitness)
