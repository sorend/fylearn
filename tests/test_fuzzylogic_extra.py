"""Additional tests for fuzzylogic functions not covered by test_fuzzylogic.py."""

import numpy as np
import pytest

import fylearn.fuzzylogic as fl


def test_trapezoidal_set():
    t = fl.TrapezoidalSet(0.0, 0.2, 0.6, 1.0)
    y = t(np.array([0.0, 0.1, 0.3, 0.5, 0.8, 1.0]))
    assert y[0] == 0.0
    assert np.isclose(y[1], 0.5)
    assert y[2] == 1.0
    assert y[3] == 1.0
    assert np.isclose(y[4], 0.5)
    assert y[5] == 0.0
    assert "T(" in str(t)


def test_trapezoidal_set_outside():
    t = fl.TrapezoidalSet(0.0, 0.2, 0.6, 1.0)
    y = t(np.array([-0.5, 1.5]))
    assert np.all(y == 0.0)


def test_zadeh_negated_set():
    tri = fl.TriangularSet(0.0, 0.5, 1.0)
    neg = fl.ZadehNegatedSet(tri)
    y = neg(np.array([0.0, 0.5, 1.0]))
    assert y[0] == 1.0
    assert np.isclose(y[1], 0.0)
    assert y[2] == 1.0
    assert "Not(" in str(neg)


def test_gowa_class():
    g = fl.GOWA(2.0, np.array([0.4, 0.6]))
    y = g(np.array([[0.2, 0.8]]), axis=1)
    expected = (0.4 * 0.8**2 + 0.6 * 0.2**2) ** 0.5
    assert np.isclose(y[0], expected)
    assert "GOWA(2.000000" in str(g)


def test_gowa_invalid_length():
    g = fl.GOWA(2.0, np.array([0.4, 0.6]))
    with pytest.raises(ValueError):
        g(np.array([0.1, 0.2, 0.3]))


def test_lukasiewicz():
    X = np.array([[0.5, 0.7], [0.1, 0.2]])
    assert np.isclose(fl.lukasiewicz_i(X)[0], 0.2)
    assert fl.lukasiewicz_i(X)[1] == 0.0
    assert np.isclose(fl.lukasiewicz_u(X)[0], 1.0)
    assert np.isclose(fl.lukasiewicz_u(X)[1], 0.3)


def test_einstein():
    X = np.array([[0.5, 0.5]])
    assert np.isclose(fl.einstein_i(X)[0], 0.2)
    assert np.isclose(fl.einstein_u(X)[0], 0.8)


def test_algebraic_sum():
    X = np.array([[0.5, 0.5]])
    assert np.isclose(fl.algebraic_sum(X, axis=1)[0], 0.75)


def test_min_max_normalize():
    X = np.array([0.0, 2.0, 4.0])
    y = fl.min_max_normalize(X)
    assert np.allclose(y, [0.0, 0.5, 1.0])


def test_dispersion():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(fl.dispersion(w), np.log(4.0))


def test_ndispersion():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(fl.ndispersion(w), 1.0)


def test_dispersion_zeros_filtered():
    w = np.array([1.0, 0.0])
    assert fl.dispersion(w) == 0.0


def test_yager_andness():
    w = np.array([1.0, 0.0])
    assert np.isclose(fl.yager_andness(w), 1.0 - fl.yager_orness(w))


def test_owa_attributes():
    o = fl.owa(0.3, 0.7)
    # v[0] is applied to the largest element, so orness is 0.3 here
    assert np.isclose(o.orness(), 0.3)
    assert np.isclose(o.andness(), 0.7)
    assert o.disp() > 0.0
    assert o.ndisp() > 0.0
    assert repr(o) == str(o)


def test_owa_sorted_mean():
    o = fl.owa(0.3, 0.7)
    X = np.array([0.2, 0.8])  # pre-sorted ascending (as np.sort produces)
    assert np.isclose(o.sorted_mean(X, axis=0), 0.3 * 0.8 + 0.7 * 0.2)


def test_owa_vector_axis_0():
    o = fl.owa(0.3, 0.7)
    X = np.array([[0.8, 0.6], [0.2, 0.4]])
    y = o(X, axis=0)
    assert y.shape == (2,)
    # columns sorted ascending then weighted with reversed v
    assert np.isclose(y[0], 0.3 * 0.8 + 0.7 * 0.2)
    assert np.isclose(y[1], 0.3 * 0.6 + 0.7 * 0.4)


def test_owa_invalid_length():
    o = fl.owa(0.3, 0.7)
    with pytest.raises(ValueError):
        o(np.array([0.1, 0.2, 0.3]))


def test_owa_single_array_arg():
    o = fl.owa(np.array([0.3, 0.7]))
    assert len(o.v) == 2


def test_meowa_edge_cases():
    # orness 0.5 -> uniform
    o = fl.meowa(4, 0.5)
    assert np.allclose(o.v, np.ones(4) / 4)
    # orness 0.0 -> all weight on last
    o = fl.meowa(4, 0.0)
    assert np.allclose(o.v, [0.0, 0.0, 0.0, 1.0])
    # orness 1.0 -> all weight on first
    o = fl.meowa(4, 1.0)
    assert np.allclose(o.v, [1.0, 0.0, 0.0, 0.0])


def test_meowa_invalid():
    with pytest.raises(ValueError):
        fl.meowa(4, 1.5)
    with pytest.raises(ValueError):
        fl.meowa(4, -0.1)
    with pytest.raises(ValueError):
        fl.meowa(1, 0.5)


def test_meowa_approximate_orness():
    for orness in [0.2, 0.8]:
        o = fl.meowa(5, orness)
        assert np.isclose(o.orness(), orness, atol=1e-4)
        assert np.isclose(np.sum(o.v), 1.0)


def test_sampling_owa_orness():
    x = np.array([0.9, 0.5, 0.2])
    o = fl.sampling_owa_orness(x, 0.5)
    assert np.isclose(np.sum(o.v), 1.0, atol=1e-2)
    # output should equal requested value d
    out = np.sum(np.sort(x)[::-1] * o.v)
    assert np.isclose(out, 0.5, atol=1e-2)


def test_sampling_owa_ndisp():
    x = np.array([0.9, 0.5, 0.2])
    o = fl.sampling_owa_ndisp(x, 0.6)
    assert np.isclose(np.sum(o.v), 1.0, atol=1e-2)
    out = np.sum(np.sort(x)[::-1] * o.v)
    assert np.isclose(out, 0.6, atol=1e-2)


def test_sampling_owa_too_short():
    with pytest.raises(ValueError):
        fl.sampling_owa_orness(np.array([0.5]), 0.5)


def test_mvowa():
    o = fl.mvowa(5, 0.7)
    assert np.isclose(np.sum(o.v), 1.0, atol=1e-2)
    assert np.isclose(o.orness(), 0.7, atol=1e-2)
    assert np.all(o.v >= 0.0)


def test_mvowa_invalid():
    with pytest.raises(ValueError):
        fl.mvowa(5, 1.5)
    with pytest.raises(ValueError):
        fl.mvowa(1, 0.5)


def test_andness_directed_averaging():
    # tnorm case (p <= 0.5): mean of x^alpha, then ^(1/alpha)
    a = fl.aa(0.3)
    X = np.array([[0.5, 0.7]])
    y = a(X, axis=1)
    assert y.shape == (1,)
    assert 0.0 <= y[0] <= 1.0
    # t-conorm case (p > 0.5)
    a = fl.aa(0.8)
    y = a(X, axis=1)
    assert 0.0 <= y[0] <= 1.0
    assert y[0] != fl.aa(0.3)(X, axis=1)[0]


def test_aa_invalid():
    with pytest.raises(AssertionError):
        fl.aa(0.0)
    with pytest.raises(AssertionError):
        fl.aa(1.0)


def test_weights_mapping_normalizes():
    w = fl.weights_mapping(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w > 0.0)


def test_helper_np_array_errors():
    with pytest.raises(ValueError):
        fl.helper_np_array({"a": 1})


def test_p_normalize_nan_rows_axis1():
    X = np.array([[0.5, 0.5], [0.0, 0.0]])
    y = fl.p_normalize(X, axis=1)
    assert np.allclose(y[0], [0.5, 0.5])
    assert np.allclose(y[1], [0.5, 0.5])  # zero row gets uniform
