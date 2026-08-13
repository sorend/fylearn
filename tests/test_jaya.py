import numpy as np
import pytest

from fylearn.ga import helper_n_generations
from fylearn.jaya import JayaOptimizer


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


def test_jaya_best_returns_tuple():
    f = lambda x: np.var(x)
    j = JayaOptimizer(f, np.zeros(3), np.ones(3), n_population=10, random_state=1)
    best_x, best_fit = j.best()
    assert best_x.shape == (3,)
    assert isinstance(best_fit, float)


def test_jaya_next_alias():
    f = lambda x: np.var(x)
    j = JayaOptimizer(f, np.zeros(3), np.ones(3), n_population=10, random_state=1)
    initial = j.best()[1]
    j.next()
    assert len(j.bestcosts_) == 2
    assert j.best()[1] <= initial


def test_jaya_respects_bounds():
    f = lambda x: np.sum(x**2)
    lb = np.array([-0.5, -0.5])
    ub = np.array([0.5, 0.5])
    j = JayaOptimizer(f, lb, ub, n_population=20, random_state=2)
    for _ in range(10):
        next(j)
    assert np.all(j.population_ >= lb)
    assert np.all(j.population_ <= ub)
    assert j.best()[1] <= 0.5
