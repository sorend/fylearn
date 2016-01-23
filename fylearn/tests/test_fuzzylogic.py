
import numpy as np
import fylearn.fuzzylogic as fl
import pytest

def test_max():
    a = np.array([0.0, 0.5, 0.4, 0.2])

    m = fl.max(a)
    assert m == 0.5

    b = np.array([[0.3, 0.2, 0.5],
                  [0.1, 0.4, 0.4]])

    m = fl.max(b, None)
    assert m == 0.5

    m = fl.max(b, 1)  # axis=1 -> row-wise
    assert len(m) == 2
    assert m[0] == 0.5
    assert m[1] == 0.4

    m = fl.max(b, 0)
    assert len(m) == 3
    assert m[0] == 0.3
    assert m[1] == 0.4
    assert m[2] == 0.5

def test_min():
    a = np.array([0.0, 0.5, 0.4, 0.2])

    m = fl.min(a)
    assert m == 0.0

    b = np.array([[0.3, 0.2, 0.5],
                  [0.1, 0.4, 0.4]])

    m = fl.min(b, None)
    assert m == 0.1

    m = fl.min(b, 1)  # axis=1 -> row-wise
    assert len(m) == 2
    assert m[0] == 0.2
    assert m[1] == 0.1

    m = fl.min(b, 0)
    assert len(m) == 3
    assert m[0] == 0.1
    assert m[1] == 0.2
    assert m[2] == 0.4


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

def test_yager_orness():

    def almost(a, b, prec=0.00001):
        return np.abs(a - b) < prec

    w = np.array([1.0, 0, 0, 0, 0, 0])
    assert almost(1.0, fl.yager_orness(w))
    assert almost(0.0, fl.yager_andness(w))

    w = np.array([1.0, 0.0])
    assert almost(1.0, fl.yager_orness(w))
    assert almost(0.0, fl.yager_andness(w))

    w = np.array([0.0, 0.0, 0.0, 1.0])
    assert almost(0.0, fl.yager_orness(w))
    assert almost(1.0, fl.yager_andness(w))

    w = np.array([0.0, 1.0])
    assert almost(0.0, fl.yager_orness(w))
    assert almost(1.0, fl.yager_andness(w))

    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert almost(0.5, fl.yager_orness(w))
    assert almost(0.5, fl.yager_andness(w))

    w = np.array([0.25, 0.25, 0.25, 0.25])
    assert almost(0.5, fl.yager_orness(w))
    assert almost(0.5, fl.yager_andness(w))

def test_owa():

    def almost(a, b, prec=0.00001):
        return np.abs(a - b) < prec

    X = np.array([1.0, 1.0, 1.0])
    w = [0.5, 0.3, 0.2]

    owa = fl.owa(w)

    r = owa(X)

    assert 1.0 == r

    owa = fl.owa(0.5, 0.3, 0.2)
    r2 = owa(X)

    assert r == r2

    with pytest.raises(ValueError) as v:
        owa = fl.owa(0.5, 0.3, 0.2)
        owa(np.array([0, 1, 0.5, 0.2, 0.3]))

    assert "len(X) != len(v)" in str(v.value)

    with pytest.raises(ValueError) as v:
        owa = fl.owa(0.5, 0.3, 0.2, 0.4)
        owa(np.array([0, 1, 0.5]))

    assert "len(X) != len(v)" in str(v.value)

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

def test_p_normalize():

    def almost(a, b):
        return np.abs(a - b) < 0.00001

    X = np.array([1, 2, 3, 4])

    Y = fl.p_normalize(X)

    assert almost(1. / 10, Y[0])
    assert almost(2. / 10, Y[1])
    assert almost(3. / 10, Y[2])
    assert almost(4. / 10, Y[3])

    X = np.array([
        [1, 2, 3, 4],
        [4, 4, 2, 0],
    ])

    Y = fl.p_normalize(X, 1)

    assert almost(1. / 10, Y[0][0])
    assert almost(2. / 10, Y[0][1])
    assert almost(3. / 10, Y[0][2])
    assert almost(4. / 10, Y[0][3])

    assert almost(4. / 10, Y[1][0])
    assert almost(4. / 10, Y[1][1])
    assert almost(2. / 10, Y[1][2])
    assert almost(0. / 10, Y[1][3])

    X = np.array([
        [3, 9],
        [3, 1],
    ])

    Y = fl.p_normalize(X, 0)

    assert 0.5 == Y[0][0]
    assert 0.5 == Y[1][0]

    assert 0.9 == Y[0][1]
    assert 0.1 == Y[1][1]

def test_p_normalize_zero_row():

    # no axis given
    X = np.array([
        [0, 0, 0],
        [0, 0, 0]
    ])
    Y = fl.p_normalize(X)

    assert Y.shape == (2, 3)
    assert np.sum(Y) == 0.0

    X = np.array([
        [0.0, 0.0, 0.0],
        [0, 0, 0]
    ])

    Y = fl.p_normalize(X, 1)

    assert Y.shape == (2, 3)
    assert np.sum(Y) == 0.0

    X = np.array([
        [0.0, 0.0, 0.0],
        [0, 0, 0],
    ])

    Y = fl.p_normalize(X, 0)

    assert Y.shape == (2, 3)
    assert np.sum(Y) == 0.0

def test_p_normalize_wrong_dimensions():

    with pytest.raises(ValueError):
        X = np.array([[1, 2, 3]])
        Y = fl.p_normalize(X, 2)

    with pytest.raises(ValueError):
        X = np.array([[1, 2, 3]])
        Y = fl.p_normalize(X, -1)

def test_meowa():

    def almost(a, b, prec=0.00001):
        return np.abs(a - b) < prec

    m = fl.meowa(5, 0.5)

    assert m is not None
    assert len(m.v) == 5
    for i in range(5):
        assert almost(0.2, m.v[i])
    assert almost(0.5, m.andness())
    assert almost(0.5, m.orness())

    m = fl.meowa(3, 0.5)

    assert m is not None
    assert len(m.v) == 3
    for i in range(3):
        assert almost(0.33, m.v[i], 0.01)
    assert almost(0.5, m.andness())

    m = fl.meowa(3, 0.8)  # example from paper

    assert m is not None
    assert len(m.v) == 3

    print "m", m.v

    assert almost(0.08187, m.v[0], 0.00001)
    assert almost(0.23627, m.v[1], 0.00001)
    assert almost(0.68187, m.v[2], 0.00001)
    assert almost(0.8, m.andness())
    assert almost(0.2, m.orness())

    m = fl.meowa(3, 1.0)
    assert len(m.v) == 3
    assert almost(0.0, m.v[0])
    assert almost(0.0, m.v[1])
    assert almost(1.0, m.v[2])
    assert almost(1.0, m.andness())

    m = fl.meowa(4, 0.0)
    assert len(m.v) == 4
    assert almost(1.0, m.v[0])
    assert almost(0.0, m.v[1])
    assert almost(0.0, m.v[2])
    assert almost(0.0, m.v[3])
    assert almost(0.0, m.andness())

    with pytest.raises(ValueError) as v:
        fl.meowa(1, 0.3)

    assert "n must be > 1" in str(v.value)

    with pytest.raises(ValueError) as v:
        fl.meowa(4, -0.5)

    assert "andness must be" in str(v.value)

    with pytest.raises(ValueError) as v:
        fl.meowa(1, 1.5)

    assert "andness must be" in str(v.value)

def test_meowa_many_decimals():

    with pytest.raises(ValueError) as v:
        m = fl.meowa(n=34, andness=0.91962632217914819)  # cant solve this.

    m = fl.meowa(n=34, andness=0.91962632217914819, maxiter=1000)  # can with more iter

    def almost(a, b, prec=0.00001):
        return np.abs(a - b) < prec

    assert len(m.v) == 34
    assert almost(0.91962, m.andness(), 0.00001)
