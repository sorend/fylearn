from __future__ import print_function

import numpy as np
from sklearn.datasets import load_iris

import fylearn.fpcga as fpcga

import pytest

# def test_classifier():

#     l = fpcga.FuzzyPatternClassifierLGA(iterations=8, epsilon=None)

#     X = np.array([
#         [0.1, 0.2, 0.4],
#         [0.11, 0.3, 0.5],
#         [0.07, 0.18, 0.38],
#         [0.2, 0.4, 0.8],
#         [0.18, 0.42, 0.88],
#         [0.22, 0.38, 0.78],
#     ])

#     y = np.array([
#         1,
#         1,
#         1,
#         0,
#         0,
#         0,
#     ])

#     l.fit(X, y)

#     print("protos_", l.protos_)

#     y_pred = l.predict([[0.0, 0.3, 0.35],
#                         [0.1, 0.4, 0.78]])

#     print("y_pred", y_pred)

#     assert len(y_pred) == 2
#     assert y_pred[0] == 1
#     assert y_pred[1] == 0


def test_classifier_iris():

    iris = load_iris()

    X = iris.data
    y = iris.target

    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)

    l = fpcga.FuzzyPatternClassifierGA(iterations=100, random_state=1)

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(l, X, y, cv=10)

    assert len(scores) == 10
    assert np.mean(scores) > 0.6
    mean = np.mean(scores)

    print("mean", mean)

    assert 0.92 == pytest.approx(mean, 0.01)
