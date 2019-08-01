from __future__ import print_function
import numpy as np

from fylearn.fuzzylogic import TriangularSet
from fylearn.nonstationary import helper_stationary_value, NonstationaryFuzzySet


def test_simple_stationary():
    """
    Constructs a nonstationary fuzzy set, with all stationary values.
    """

    s = NonstationaryFuzzySet(TriangularSet,
                              a=helper_stationary_value(1),
                              b=helper_stationary_value(2),
                              c=helper_stationary_value(3))

    T = 0.
    X = [1., 2., 3.]
    Y = s(T, X)

    assert 0 == Y[0][0]
    assert 1 == Y[0][1]
    assert 0 == Y[0][2]

def test_simple():
    """
    Constructs a nostationary fuzzy set based on a decreasing and increasing behaviour.
    """

    def helper_decreasing(val):
        return lambda t: val - t

    def helper_increasing(val):
        return lambda t: val + t

    s = NonstationaryFuzzySet(TriangularSet,
                              a=helper_decreasing(1),
                              b=helper_decreasing(2),
                              c=helper_increasing(3))

    T = [0., 1.]
    X = [
        [0., 1., 2., 3., 4.],
        [0., 1., 2., 3., 4.]
    ]

    Y = s(T, X)

    print("Y", Y)

    assert 2 == Y.ndim

    assert 0 == Y[0][0]
    assert 0 == Y[0][1]
    assert 1 == Y[0][2]
    assert 0 == Y[0][3]
    assert 0 == Y[0][4]

    # set = T(0, 1, 4)
    assert 0 == Y[1][0]
    assert 1 == Y[1][1]

    diff = 0.67 - Y[1][2]
    assert diff < 0.01
    diff = 0.33 - Y[1][3]
    assert diff < 0.01

    assert 0 == Y[1][4]

def test_n_dim():

    s = NonstationaryFuzzySet(TriangularSet,
                              a=helper_stationary_value(1.),
                              b=helper_stationary_value(2.),
                              c=helper_stationary_value(3.))

    T = [0, 1, 2, 3, 4]
    X = [
        [
            [1., 2.], [2., 3.],
            [3., 4.], [4., 5.]
        ],
        [
            [3., 4.], [4., 5.],
            [1., 2.], [2., 3.]
        ],
        [
            [1., 2.], [2., 3.],
            [1., 2.], [2., 3.],
        ],
        [
            [3., 4.], [4., 5.],
            [3., 4.], [4., 5.]
        ],
        [
            [1., 2.], [2., 3.],
            [1., 2.], [2., 3.]
        ]
        ]
    Y = s(T, X)

    assert len(Y) == len(T)
    assert Y.ndim == 3
    assert len(Y[0]) == len(X[0])

def test_paper():

    def f_c(t):
        if t in (1, 3, 5):
            return 0.
        elif t == 2:
            return -2.
        elif t == 4:
            return 2.
        else:
            raise ValueError("Only defined for t in {1, 2, 3, 4, 5}")

    def f_rho(t):
        if t in (1, 2, 4):
            return -1.
        elif t == 3:
            return 0.
        elif t == 5:
            return 1.
        else:
            raise ValueError("Only defined for t in {1, 2, 3, 4, 5}")

    class GaussianSample:
        def __init__(self, c, rho):
            self.c = c
            self.rho = rho

        def __call__(self, X):
            return np.exp(-(((X - self.c) ** 2) / ((2 * self.rho) ** 2)))

    s = NonstationaryFuzzySet(GaussianSample, c=f_c, rho=f_rho)

    T = [1, 2, 3, 4, 5]
    X = [ range(100), range(100), range(100), range(100), range(100) ]

    Y = s(T, X)
    print("Y", Y)
