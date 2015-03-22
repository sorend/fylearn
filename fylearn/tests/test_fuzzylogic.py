
import numpy as np
from sklearn.utils.testing import assert_equal, assert_true, assert_almost_equal

import fylearn.fuzzylogic as fl

def test_owa():

    X = np.array([1.0, 1.0, 1.0, 1.0])
    w = [0.5, 0.3, 0.2]

    owa = fl.owa(w)

    r = owa(X)

    assert_equal(1.0, r)

    owa = fl.owa(0.5, 0.3, 0.2)
    r2 = owa(X)

    assert_equal(r, r2)

def test_owa_matrix():

    X = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    w = [0.5, 0.3, 0.2]

    owa = fl.owa(w)

    r = owa(X)

    assert_equal(1.0, r[0])
    assert_equal(0.5, r[1])

def test_aa_tco():

    h = fl.aa(0.2)

    assert_equal(1, h([1.0, 1.0, 1.0, 1.0]))

    assert_equal(0, h([0.0, 0.0, 0.0, 0.0]))

    # 0.2, 0.6, 0.7, 0.9
    # p = 0.2
    # a = (1.0 - p) / p
    # a = 4
    # (((0.2**a) + (0.6**a) + (0.7**a) + (0.9**a)) / 4.0)**(1.0/a)
    # = 0.7119
    assert_almost_equal(0.7119, h([0.2, 0.6, 0.7, 0.9]), 4)

def test_aa_t():

    h = fl.aa(0.8)

    assert_equal(1, h([1.0, 1.0, 1.0, 1.0]))

    assert_equal(0, h([0.0, 0.0, 0.0, 0.0]))

    # 0.2, 0.6, 0.7, 0.9
    # p = 0.8
    # a = p / (1.0 - p)
    # a = 4
    # 1.0 - ((((1.0-0.2)**(1.0/a)) + ((1.0-0.6)**(1.0/a)) + ((1.0-0.7)**(1.0/a)) + ((1.0-0.9)**(1.0/a))) / 4.0)**a
    # = 0.6649
    assert_almost_equal(0.6649, h([0.2, 0.6, 0.7, 0.9]), 4)
