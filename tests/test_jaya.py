from __future__ import print_function
import numpy as np
from fylearn.jaya import JayaOptimizer
from fylearn.ga import helper_n_generations
import pytest

def test_jaya_variance():

    lb = np.zeros(5)
    lb[::2] = 1.1

    ub = np.ones(5)
    ub[::2] = 2.2

    o = JayaOptimizer(f=lambda x: np.abs(np.mean(x) - 0.5), lower_bound=lb, upper_bound=ub)

    o = helper_n_generations(o, 20)

    print("costs history", o.bestcosts_)
    solution, fitness = o.best()
    print("best fitness", fitness)
    print("best solution", solution)

    assert len(o.fitness_) == 50
    assert len(o.population_) == 50
    assert len(o.bestcosts_) == 21

def test_jaya_sphere():
    """Example given in paper"""

    o = JayaOptimizer(f=lambda x: np.sum(x**2),
                      lower_bound=np.ones(10) * -10.0,
                      upper_bound=np.ones(10) * 10.0,
                      n_population=34)

    o = helper_n_generations(o, 100)

    solution, fitness = o.best()

    print("costs history", o.bestcosts_)
    print("best fitness", fitness)
    print("best solution", solution)

    assert len(o.fitness_) == 34
    assert len(o.population_) == 34
    assert len(o.bestcosts_) == 101



def test_jaya_random_state_wrong_value():
    """Example given in paper"""

    params = {'f':lambda x: np.sum(x**2),
             'lower_bound':np.ones(10) * -10.0,
             'upper_bound':np.ones(10) * 10.0,
             'n_population':34,
             'random_state':'wrong_value'
    }

    with pytest.raises(ValueError):
        JayaOptimizer(**params)


def test_jaya_sphere_bounds():
    """ Another example with strange domain """

    o = JayaOptimizer(f=lambda x: np.sum(x**2),
                      lower_bound=np.array([1, 0.001, 100]),
                      upper_bound=np.array([10, 0.2, 1000]),
                      n_population=34)

    o = helper_n_generations(o, 100)

    solution, fitness = o.best()

    print("costs history", o.bestcosts_)
    print("best fitness", fitness)
    print("best solution", solution)

    assert len(o.fitness_) == 34
    assert len(o.population_) == 34
    assert len(o.bestcosts_) == 101
