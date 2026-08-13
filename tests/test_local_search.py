
import logging

import numpy as np

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


def test_ps_optimize_step_improves():
    from sklearn.utils import check_random_state

    from fylearn.local_search import ps_optimize_step

    rs = check_random_state(0)
    f = lambda x: np.sum(x**2)
    x = np.array([-1.0, -1.0])
    d = np.array([0.5, 0.5])
    _, new_fitness, _ = ps_optimize_step(f, x, d, f(x), np.array([-2.0, -2.0]), np.array([2.0, 2.0]), rs)
    assert new_fitness < 2.0


def test_ps_optimize_step_rejects_and_inverts():
    from sklearn.utils import check_random_state

    from fylearn.local_search import ps_optimize_step

    rs = check_random_state(0)
    f = lambda x: np.sum(x**2)
    x = np.array([0.0, 0.0])  # already at the minimum
    d = np.array([0.5, 0.5])
    x2, new_fitness, d2 = ps_optimize_step(f, x, d, f(x), np.array([-2.0, -2.0]), np.array([2.0, 2.0]), rs)
    # position restored, direction halved and inverted for the modified index
    assert np.allclose(x2, [0.0, 0.0])
    assert any(dd == -0.25 for dd in d2)


def test_lus_optimize_step():
    from sklearn.utils import check_random_state

    from fylearn.local_search import lus_optimize_step

    rs = check_random_state(0)
    f = lambda x: np.sum(x**2)
    x = np.array([1.0, 1.0])
    d = np.array([0.5, 0.5])
    y, new_fitness, d = lus_optimize_step(f, x, d, f(x), np.array([-2.0, -2.0]), np.array([2.0, 2.0]), rs, q=0.5)
    assert 0.0 <= new_fitness <= 4.5
    assert np.all(np.abs(y) <= 2.0)


def test_init_position_within_bounds():
    from sklearn.utils import check_random_state

    from fylearn.local_search import init_position

    rs = check_random_state(1)
    lb = np.array([-1.0, 2.0])
    ub = np.array([1.0, 4.0])
    for _ in range(20):
        x = init_position(rs, lb, ub)
        assert np.all(x >= lb)
        assert np.all(x <= ub)


def test_sample_bounded_within_bounds():
    from sklearn.utils import check_random_state

    from fylearn.local_search import sample_bounded

    rs = check_random_state(2)
    x = np.array([0.9, -0.9])
    d = np.array([0.5, 0.5])
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    for _ in range(20):
        y = sample_bounded(rs, x, d, lb, ub)
        assert np.all(y >= lb)
        assert np.all(y <= ub)


def test_scipy_refine():
    from fylearn.local_search import scipy_refine

    f = lambda x: np.sum(x**2)
    best_x, best_fitness = scipy_refine(f, np.array([0.5, 0.5]), 0.5, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert np.allclose(best_x, [0.0, 0.0], atol=1e-3)
    assert best_fitness < 0.01


def test_helper_num_runs_with_refine():
    from sklearn.utils import check_random_state

    from fylearn.local_search import helper_num_runs, scipy_refine

    opt = ls.LocalUnimodalSamplingOptimizer(
        lambda x: np.sum(x**2),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        random_state=check_random_state(0),
        max_evaluations=50,
    )
    best_x, best_fitness = helper_num_runs(opt, num_runs=5, refine=scipy_refine)
    assert np.all(np.abs(best_x) <= 1.0)
    assert best_fitness < 0.5


def test_helper_generations_bestidx():
    from sklearn.utils import check_random_state

    from fylearn.local_search import helper_generations

    opt = ls.PatternSearchOptimizer(
        lambda x: np.sum(x**2),
        np.array([-1.0, -1.0]),
        np.array([1.0, 1.0]),
        random_state=check_random_state(3),
        max_evaluations=20,
    )
    wrapped = helper_generations(opt)
    for _ in range(3):
        next(wrapped)
    assert wrapped.X_.shape[0] == 3
    idx = wrapped.bestidx(2)
    assert len(idx) == 2
    X, f = wrapped.best(2)
    assert X.shape == (2, 2)
    assert np.all(np.diff(f) >= 0)
