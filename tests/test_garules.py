from __future__ import print_function
import numpy as np

from fylearn import garules

from sklearn.datasets import load_iris
import pytest

def test_classifier():

    l = garules.MultimodalEvolutionaryClassifier(n_iterations=100)

    X = np.array([
        [1, 2, 4],
        [2, 4, 8]
    ])

    y = np.array([
        0,
        1
    ])

    l.fit(X, y)

    assert [0] == l.predict([[0.9, 1.7, 4.5]])

    assert [1] == l.predict([[2.1, 3.9, 7.8]])

def test_classifier_iris():

    iris = load_iris()

    X = iris.data
    y = iris.target

    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)

    l = garules.MultimodalEvolutionaryClassifier(n_iterations=100, random_state=1)

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(l, X, y, cv=10)
    mean = np.mean(scores)

    assert 0.93 == pytest.approx(mean, 0.01)  # using the same random state expect same


# def test_compare_diabetes():
#     import os
#     csv_file = os.path.join(os.path.dirname(__file__), "diabetes.csv")
#     data = np.genfromtxt(csv_file, dtype=float, delimiter=',', names=True)

#     X = np.array([data["preg"], data["plas"], data["pres"], data["skin"],
#                   data["insu"], data["mass"], data["pedi"], data["age"]]).T
#     y = data["class"]

#     from sklearn.preprocessing import MinMaxScaler
#     X = MinMaxScaler().fit_transform(X)

#     l = garules.MultimodalEvolutionaryClassifier(n_iterations=100)

#     from sklearn import cross_validation

#     scores = cross_validation.cross_val_score(l, X, y, cv=10)
#     mean = np.mean(scores)

#     print "mean", mean

#     assert_true(0.68 < mean)

#     from sklearn.ensemble import BaggingClassifier

#     l = BaggingClassifier(garules.MultimodalEvolutionaryClassifier(n_iterations=100))

#     scores = cross_validation.cross_val_score(l, X, y, cv=10)
#     mean = np.mean(scores)

#     print "mean", mean

#     assert_true(0.80 < mean)
