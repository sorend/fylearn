
from __future__ import print_function

import numpy as np
import pytest

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
