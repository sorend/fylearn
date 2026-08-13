

import numpy as np
import pytest

import fylearn.fuzzylogic as fl
from fylearn.frr import (
    FuzzyReductionRuleClassifier,
    build_aiwa_operator,
    build_memberships,
    build_owa_operator,
    pi_factory,
)
from fylearn.frr import ModifiedFuzzyPatternClassifier as MFPC


@pytest.mark.parametrize("D_val", [2, 4, 6, 8])
def test_mfpc_good_D(D_val):
    l = MFPC(D=D_val)
    assert l is not None
    assert isinstance(l, MFPC)


@pytest.mark.parametrize("D_val", [-1, 1, 3, 9, 1000])
def test_mfpc_bad_D(D_val):
    with pytest.raises(ValueError) as ve:
        l = MFPC(D=D_val)


@pytest.mark.parametrize("pce_val", [0.0, 0.25, 0.99, 1.0])
def test_mfpc_good_pce(pce_val):
    l = MFPC(pce=pce_val)
    assert l is not None
    assert isinstance(l, MFPC)


@pytest.mark.parametrize("pce_val", [-0.000001, 1.000001])
def test_mfpc_bad_pce(pce_val):
    with pytest.raises(ValueError) as ve:
        l = MFPC(pce=pce_val)


@pytest.mark.parametrize("operator_val", ["aiwa", "owa"])
def test_mfpc_good_operator(operator_val):
    l = MFPC(operator=operator_val)
    assert l is not None
    assert isinstance(l, MFPC)


@pytest.mark.parametrize("operator_val", ["aiwas", "gowa", "asdf", None, 0])
def test_mfpc_bad_operator(operator_val):
    with pytest.raises(ValueError) as ve:
        l = MFPC(operator=operator_val)


@pytest.mark.parametrize("andness_val", [0.5, 0.5001, 0.75, 0.99, 1.0])
def test_mfpc_good_andness(andness_val):
    l = MFPC(andness=andness_val)
    assert l is not None
    assert isinstance(l, MFPC)


@pytest.mark.parametrize("andness_val", [-0.1, 0.0, 0.49999, 1.000001, 10])
def test_mfpc_bad_andness(andness_val):
    with pytest.raises(ValueError) as ve:
        l = MFPC(andness=andness_val)


def test_classifier():

    l = MFPC()

    print("l", l.get_params())

    X = np.array([
        [0.1, 0.2, 0.4],
        [0.11, 0.3, 0.5],
        [0.2, 0.4, 0.8],
        [0.18, 0.42, 0.88]
    ])

    y = np.array([
        1,
        1,
        0,
        0
    ])

    l.fit(X, y)

    y_pred = l.predict([[0.0, 0.3, 0.35],
                        [0.1, 0.4, 0.78]])

    print("y_pred", y_pred)

    assert len(y_pred) == 2
    assert y_pred[0] == 1
    assert y_pred[1] == 0


def test_classifier_delta_zero():

    l = MFPC()

    print("l", l.get_params())

    X = np.array([
        [0.1, 0.1],
        [0.11, 0.1],
        [0.2, 0.1],
        [0.18, 0.1]
    ])

    y = np.array([
        1,
        1,
        0,
        0
    ])

    l.fit(X, y)

    y_pred = l.predict([[0.1, 0.1],
                        [0.2, 0.1]])

    print("y_pred", y_pred)

    assert len(y_pred) == 2
    assert y_pred[0] == 1
    assert y_pred[1] == 0


def test_fuzzy_reduction_rule_classifier():
    l = FuzzyReductionRuleClassifier()

    X = np.array([[0.1, 0.2], [0.12, 0.25], [0.9, 0.8], [0.85, 0.75]])
    y = np.array([1, 1, 0, 0])

    l.fit(X, y)
    y_pred = l.predict([[0.11, 0.22], [0.88, 0.79]])

    assert len(y_pred) == 2
    assert y_pred[0] == 1
    assert y_pred[1] == 0


def test_frr_predict_before_fit():
    l = FuzzyReductionRuleClassifier()
    with pytest.raises(Exception) as e:
        l.predict(np.array([[0.1, 0.2]]))
    assert "fit" in str(e.value)


def test_frr_nan_classes():
    l = FuzzyReductionRuleClassifier()
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    y = np.array([0.0, np.nan])
    with pytest.raises(Exception):
        l.fit(X, y)


def test_frr_aggregation_with_axis():
    # aggregation with axis support is preferred over apply_along_axis fallback
    l = FuzzyReductionRuleClassifier(aggregation=fl.mean)
    X = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.9, 0.8, 0.7], [0.8, 0.7, 0.6]])
    y = np.array([0, 0, 1, 1])
    l.fit(X, y)
    y_pred = l.predict([[0.15, 0.25, 0.35], [0.85, 0.75, 0.65]])
    assert y_pred[0] == 0
    assert y_pred[1] == 1


def test_frr_generic_aggregation_fallback():
    # a function without axis support falls back to apply_along_axis
    l = FuzzyReductionRuleClassifier(aggregation=lambda row: np.max(row))
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    y = np.array([0, 1])
    l.fit(X, y)
    y_pred = l.predict([[0.15, 0.25], [0.85, 0.75]])
    assert len(y_pred) == 2


def test_mfpc_fit_returns_self():
    l = MFPC()
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    y = np.array([0, 1])
    out = l.fit(X, y)
    assert out is l


def test_mfpc_owa_operator():
    l = MFPC(operator="owa", andness=0.75)
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    y = np.array([0, 1])
    l.fit(X, y)
    assert isinstance(l.operator_, fl.OWA)
    y_pred = l.predict([[0.1, 0.2], [0.9, 0.8]])
    assert y_pred[0] == 0
    assert y_pred[1] == 1


def test_mfpc_aiwa_operator():
    l = MFPC(operator="aiwa")
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    y = np.array([0, 1])
    l.fit(X, y)
    assert isinstance(l.operator_, fl.AndnessDirectedAveraging)


def test_build_owa_operator():
    o = build_owa_operator(0.75, 3)
    assert isinstance(o, fl.OWA)
    assert len(o.v) == 3
    assert np.isclose(np.sum(o.v), 1.0)


def test_build_aiwa_operator():
    a = build_aiwa_operator(0.75, 3)
    assert isinstance(a, fl.AndnessDirectedAveraging)


def test_pi_factory():
    p = pi_factory(0.0, 0.5, 1.0)
    assert isinstance(p, fl.PiSet)


def test_build_memberships():
    X = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
    mus = build_memberships(X, pi_factory)
    assert len(mus) == 2
    assert all(isinstance(m, fl.PiSet) for m in mus)


def test_mfpc_params_set_get():
    l = MFPC(D=4, pce=0.5, andness=0.8, operator="owa")
    params = l.get_params()
    assert params == {"D": 4, "pce": 0.5, "andness": 0.8, "operator": "owa"}
    l2 = MFPC()
    l2.set_params(**params)
    assert l2.D == 4
    assert l2.pce == 0.5
    assert l2.andness == 0.8
    assert l2.operator == "owa"
