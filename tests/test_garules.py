import numpy as np
import pytest
from sklearn.datasets import load_iris

from fylearn import garules


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

    assert 0.9 < mean


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


def test_ensemble_set_params_works():
    from fylearn.garules import EnsembleMultimodalEvolutionaryClassifier

    # regression: set_params used to call self.setattr which does not exist
    l = EnsembleMultimodalEvolutionaryClassifier()
    out = l.set_params(n_iterations=5, sample_size=20)
    assert out is l
    assert l.n_iterations == 5
    assert l.sample_size == 20


def test_ensemble_nan_classes_rejected():
    from fylearn.garules import EnsembleMultimodalEvolutionaryClassifier

    l = EnsembleMultimodalEvolutionaryClassifier(n_iterations=2, random_state=1)
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([0.0, 1.0, np.nan])
    with pytest.raises(ValueError):
        l.fit(X, y)


def test_stoean_distance():
    from fylearn.garules import StoeanDistance

    d = StoeanDistance(np.array([1.0, 1.0]))
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    R = d.pairwise(X)
    assert R.shape == (2, 2)
    assert R[0][0] == 0.0
    assert R[0][1] == 2.0
    assert R[1][1] == 0.0
    # pairwise with explicit Y
    Y = np.array([[0.5, 0.5]])
    R2 = d.pairwise(X, Y)
    assert R2.shape == (2, 1)


def test_distancemetric_f():
    from fylearn.garules import distancemetric_f

    f = distancemetric_f("euclidean")
    m = f(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert m is not None


def test_multimodal_predict_proba():
    from fylearn.garules import MultimodalEvolutionaryClassifier

    l = MultimodalEvolutionaryClassifier(n_iterations=10, random_state=1)
    X = np.array([[1, 2, 4], [2, 4, 8], [1.2, 2.4, 4.8], [2.1, 3.9, 7.8]])
    y = np.array([0, 1, 0, 1])
    l.fit(X, y)
    P = l.predict_proba(X)
    assert P.shape == (len(X), 2)
    assert np.all(P >= 0.0)
    assert np.all(P <= 1.0)


def test_multimodal_distance_sum():
    from fylearn.garules import MultimodalEvolutionaryClassifier

    l = MultimodalEvolutionaryClassifier(n_iterations=1, random_state=1)
    X = np.array([[1, 2], [2, 4]])
    y = np.array([0, 1])
    l.fit(X, y)
    d = l.distance_sum(X, X)
    assert d.shape == (2,)
    assert np.allclose(d, [2.0, 2.0])


def test_ensemble_predict_proba():
    from fylearn.garules import EnsembleMultimodalEvolutionaryClassifier

    l = EnsembleMultimodalEvolutionaryClassifier(n_iterations=5, random_state=1)
    X = np.array([[1, 2, 4], [2, 4, 8], [1.2, 2.4, 4.8], [2.1, 3.9, 7.8]])
    y = np.array([0, 1, 0, 1])
    l.fit(X, y)
    P = l.predict_proba(X)
    assert P.shape == (len(X), 2)
    assert np.all(P >= 0.0)


def test_ensemble_weights_use_all_genes():
    # regression: the trained weight vector has n_models * n_classes genes and
    # all of them are now used for prediction
    from fylearn.garules import EnsembleMultimodalEvolutionaryClassifier

    l = EnsembleMultimodalEvolutionaryClassifier(n_iterations=5, n_models=3, random_state=1)
    X = np.array([[1, 2], [2, 4], [1.2, 2.4], [2.1, 3.9]])
    y = np.array([0, 1, 0, 1])
    l.fit(X, y)
    assert l.weights_.shape[0] == 3 * 2
