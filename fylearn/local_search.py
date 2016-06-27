####################################################################
#
# Pattern Search (PS) and Local Unimodal Sampling (LUS).
# Heuristic optimization methods for real-valued functions.
#
# This is a preliminary version that needs a few features before release.
#
# Reference:
#
# Tuning & Simplifying Heuristical Optimization, PhD Thesis,
# M.E.H. Pedersen, University of Southampton, 2010.
#
####################################################################

import numpy as np
import logging

from sklearn.utils import check_random_state

logger = logging.getLogger(__name__)

def init_position(rs, lower, upper):
    """
    Initialize a position in the search-space with random uniform values between
    the lower and upper boundaries.

    :param lower: Array with lower boundary for sampling range.
    :param upper: Array with upper boundary for sampling range.
    :return: Random value between lower and upper boundaries.
    """
    return rs.rand(lower.shape[0]) * (upper - lower) + lower

def sample_bounded(rs, x, d, lower_bound, upper_bound):
    """
    Generate a random sample between x-d and x+d, while ensuring the range
    is bounded by lower_bound and upper_bound.

    :param rs: A random state.
    :param x: Take a sample around this position in the search-space.
    :param d: Sample within distance d of x.
    :param lower_bound: Array with lower boundary for the search-space.
    :param upper_bound: Array with upper boundary for the search-space.
    :return:
    """
    # Adjust sampling range so it does not exceed search-space bounds.
    l = np.maximum(x - d, lower_bound)
    u = np.minimum(x + d, upper_bound)
    # Return a random sample.
    return init_position(rs, l, u)

def ps_optimize_step(f, x, d, current_fitness, lower_bound, upper_bound, rs, **kwargs):
    idx = rs.randint(0, x.shape[0])  # pick index to modify
    t = x[idx]  # save old value and update
    x[idx] = min(upper_bound[idx], max(lower_bound[idx], x[idx] + d[idx]))
    new_fitness = f(x)

    # logger.info("ps fitness %s = %.4f" % (str(x), new_fitness))

    if new_fitness < current_fitness:
        # If improvement to fitness, update just the fitness, position is already updated.
        return x, new_fitness, d
    else:
        # Reduce search-range and invert direction for this idx.
        x[idx] = t
        d[idx] *= -0.5
        return x, new_fitness, d

def lus_optimize_step(f, x, d, current_fitness, lower_bound, upper_bound, rs, **kwargs):
    """
    Perform one optimization run using the Local Unimodal Sampling (LUS) method.
    """
    # Derived control parameter for this optimizer.
    q = kwargs["q"]
    # Sample new position y from the bounded surroundings
    # of the current position x.
    y = sample_bounded(rs, x, d, lower_bound, upper_bound)
    new_fitness = f(y)

    # logger.info("lus fitness %s = %.4f" % (str(y), new_fitness))

    if new_fitness < current_fitness:
        return y, new_fitness, d
    else:
        # Otherwise decrease the search-range.
        d *= q
        return x, new_fitness, d

def scipy_refine(f, best_x, best_fitness, lower_bound, upper_bound):
    """
    Scipy refine function. Requires scipy.optimize is installed before use
    """
    import scipy.optimize
    # Refine best result using scipy (slow).
    # Scipy requires bounds in another format.
    bounds = list(zip(lower_bound, upper_bound))
    # Start scipy optimization at best found solution.
    res = scipy.optimize.minimize(fun=f, x0=best_x, method="L-BFGS-B", bounds=bounds)
    # Get best fitness and parameters.
    best_fitness = res.fun
    best_x = res.x
    # log best
    logger.info("Best Fitness: %.4f" % (best_fitness,))

    return best_x, best_fitness

class helper_generations(object):
    """
    This helper wraps the optimizer, so ga.helper_n_generations can be used for
    iteratively evaluating the optimizer much like a population based method.

    Example:
    --------

    >>> import numpy as np
    >>> from ga import helper_n_generations
    >>> f = lambda x: np.var(x)
    >>> lb, ub = np.ones(5) * -10, np.ones(5) * 10
    >>> ps = PatternSearchOptimizer(f, lower_bound=lb, upper_bound=ub)
    >>> wrapped = helper_generations(ps)
    >>> wrapped = helper_n_generations(wrapped, 100)
    >>> wrapped.best(5)  # get 5 best solutions and their fitness
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.X_ = np.zeros((0, optimizer.lower_bound.shape[0]))
        self.fitness_ = np.zeros((0,))

    def next(self):
        """Next generation"""
        X, fitness = self.optimizer()
        self.X_ = np.append(self.X_, [ X ], axis=0)
        self.fitness_ = np.append(self.fitness_, fitness)

    def bestidx(self, num=5):
        """
        Returns the indexes of the num best solutions.
        """
        return np.argsort(self.fitness_)[:num]

    def best(self, num=5):
        """
        Returns best solutions and their fitness

        Parameters:
        -----------

        num : Top-num best solutions.

        """
        idx = self.bestidx(num)
        return self.X_[idx], self.fitness_[idx]

def helper_num_runs(optimizer, num_runs=100, refine=None):
    """
    This is a helper function to evaluate an optimizer a specified number
    of times and return the best fitness and solution.
    """
    X = np.ones((num_runs, optimizer.lower_bound.shape[0]))
    fitness = np.ones(num_runs)

    for run in range(num_runs):
        X[run, :], fitness[run] = optimizer()

    best_idx = np.argmin(fitness)
    best_x, best_fitness = X[best_idx], fitness[best_idx]

    if refine is not None:  # refine function was given
        best_x, best_fitness = refine(optimizer.f, best_x, best_fitness,
                                      optimizer.lower_bound, optimizer.upper_bound)

    return best_x, best_fitness

class BaseOptimizer(object):

    def __init__(self, f, lower_bound, upper_bound, lower_init=None, upper_init=None,
                 random_state=None, max_evaluations=100):

        self.f = f

        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.lower_init = np.array(lower_bound) if lower_init is None else np.array(lower_init)
        self.upper_init = np.array(upper_bound) if upper_init is None else np.array(upper_init)

        self.random_state = check_random_state(random_state)

        self.max_evaluations = int(max_evaluations)

        self.optimize_function_args = {}

    def __call__(self):

        d = self.upper_bound - self.lower_bound  # initialize
        x = init_position(self.random_state, self.lower_init, self.upper_init)
        fitness = self.f(x)
        for evaluation in range(self.max_evaluations):
            x, new_fitness, d = self.optimize_function(self.f, x, d, fitness,
                                                       self.lower_bound, self.upper_bound,
                                                       self.random_state,
                                                       **self.optimize_function_args)
            if new_fitness < fitness:
                fitness = new_fitness

        return x, fitness

class LocalUnimodalSamplingOptimizer(BaseOptimizer):

    def __init__(self, *args, **kwargs):
        if "gamma" in kwargs:
            self.gamma = kwargs.pop("gamma")
        else:
            self.gamma = 3.0
        super(LocalUnimodalSamplingOptimizer, self).__init__(*args, **kwargs)
        self.optimize_function = lus_optimize_step
        self.q = 0.5 ** (1.0 / (self.gamma * self.lower_bound.shape[0]))  # set Q
        self.optimize_function_args["q"] = self.q

class PatternSearchOptimizer(BaseOptimizer):

    def __init__(self, *args, **kwargs):
        super(PatternSearchOptimizer, self).__init__(*args, **kwargs)
        self.optimize_function = ps_optimize_step
