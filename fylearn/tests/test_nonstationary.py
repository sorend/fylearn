import numpy as np

from fylearn.fuzzylogic import *
from fylearn.nonstationary import *
from sklearn.utils.testing import assert_equal, assert_true, assert_almost_equal


def test_simple_stationary():

    s = NonstationaryFuzzySet(TriangularSet,
                              a=helper_stationary_value(1),
                              b=helper_stationary_value(2),
                              c=helper_stationary_value(3))

    T = 0.
    X = [1., 2., 3.]
    Y = s(T, X)

    assert_equal(0, Y[0][0])
    assert_equal(1, Y[0][1])
    assert_equal(0, Y[0][2])

def test_simple():

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

    print "Y", Y

    assert_equal(2, Y.ndim)

    assert_equal(0, Y[0][0])
    assert_equal(0, Y[0][1])
    assert_equal(1, Y[0][2])
    assert_equal(0, Y[0][3])
    assert_equal(0, Y[0][4])

    # set = T(0, 1, 4)
    assert_equal(0, Y[1][0])
    assert_equal(1, Y[1][1])
    assert_almost_equal(0.67, Y[1][2], 2)
    assert_almost_equal(0.33, Y[1][3], 2)
    assert_equal(0, Y[1][4])

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

    assert_equal(len(Y), len(T))
    assert_equal(Y.ndim, 3)
    assert_equal(len(Y[0]), len(X[0]))

