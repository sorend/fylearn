
import numpy as np
import fylearn.fuzzylogic as fl

def test_helper_np_array():

    a = fl.helper_np_array(1.0)
    assert len(a) == 1
    assert a[0] == 1.0

    a = fl.helper_np_array(432)
    assert len(a) == 1
    assert a[0] == 432

    a = fl.helper_np_array((123, 4.5))
    assert len(a) == 2
    assert a[0] == 123
    assert a[1] == 4.5

    a = fl.helper_np_array([234, 5.6, 78])
    assert len(a) == 3
    assert a[0] == 234
    assert a[1] == 5.6
    assert a[2] == 78

    a = fl.helper_np_array([[1, 2], [3, 4]])
    assert len(a) == 2 and len(a[0]) == 2 and len(a[1]) == 2

    X = np.array(a)
    Y = fl.helper_np_array(X)
    assert X is Y


def test_piset():

    s = fl.PiSet(r=0.5, p=0.2, q=0.8)

    print "y", s([0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])

    assert s(0.19) < 0.5
    assert abs(s(0.2) - 0.5) < 0.0001
    assert s(0.45) > 0.5 and s(0.45) < 1.0
    assert abs(s(0.5) - 1.0) < 0.0001
    assert s(0.65) > 0.5 and s(0.65) < 1.0
    assert abs(s(0.8) - 0.5) < 0.0001
    assert s(0.81) < 0.5

def test_owa():

    X = np.array([1.0, 1.0, 1.0, 1.0])
    w = [0.5, 0.3, 0.2]

    owa = fl.owa(w)

    r = owa(X)

    assert 1.0 == r

    owa = fl.owa(0.5, 0.3, 0.2)
    r2 = owa(X)

    assert r == r2

def test_owa_matrix():

    X = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    w = [0.5, 0.3, 0.2]

    owa = fl.owa(w)

    r = owa(X)

    assert 1.0 == r[0]
    assert 0.5 == r[1]

def test_aa_tco():

    h = fl.aa(0.2)

    assert 1 == h([1.0, 1.0, 1.0, 1.0])

    assert 0 == h([0.0, 0.0, 0.0, 0.0])

    # 0.2, 0.6, 0.7, 0.9
    # p = 0.2
    # a = (1.0 - p) / p
    # a = 4
    # (((0.2**a) + (0.6**a) + (0.7**a) + (0.9**a)) / 4.0)**(1.0/a)
    # = 0.7119
    diff = 0.7119 - h([0.2, 0.6, 0.7, 0.9])
    assert diff < 0.0001

def test_aa_t():

    h = fl.aa(0.8)

    assert 1 == h([1.0, 1.0, 1.0, 1.0])

    assert 0 == h([0.0, 0.0, 0.0, 0.0])

    # 0.2, 0.6, 0.7, 0.9
    # p = 0.8
    # a = p / (1.0 - p)
    # a = 4
    # 1.0 - ((((1.0-0.2)**(1.0/a)) + ((1.0-0.6)**(1.0/a)) + ((1.0-0.7)**(1.0/a)) + ((1.0-0.9)**(1.0/a))) / 4.0)**a
    # = 0.6649
    diff = 0.6649 - h([0.2, 0.6, 0.7, 0.9])
    assert diff < 0.0001
