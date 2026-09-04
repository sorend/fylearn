import numpy as np
import pytest

from fylearn.nf_rvfl import NeuroFuzzyRVFLClassifier, NeuroFuzzyRVFLRegressor


def test_nf_rvfl_classifier_is_deterministic_and_normalizes_memberships():
    X = np.array([[-2.0, -1.0], [-1.5, -0.5], [1.0, 1.5], [2.0, 2.5]])
    y = np.array(["left", "left", "right", "right"])
    kwargs = {
        "n_fuzzy_nodes": 2,
        "n_hidden_nodes": 8,
        "cluster_method": "kmeans",
        "random_state": 7,
    }

    first = NeuroFuzzyRVFLClassifier(**kwargs).fit(X, y)
    second = NeuroFuzzyRVFLClassifier(**kwargs).fit(X, y)

    assert np.array_equal(first.predict(X), second.predict(X))
    assert np.allclose(first.predict_proba(X).sum(axis=1), 1.0)
    assert np.allclose(first._memberships(X).sum(axis=1), 1.0)
    assert first.predict_proba(X).shape == (4, 2)


def test_nf_rvfl_regressor_supports_multioutput_and_random_centers():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.column_stack((X[:, 0] - X[:, 1], X[:, 2] ** 2))
    model = NeuroFuzzyRVFLRegressor(
        n_fuzzy_nodes=3,
        n_hidden_nodes=5,
        cluster_method="random",
        C=1.0,
        random_state=3,
    ).fit(X, y)

    prediction = model.predict(X)
    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()
    assert model.centers_.shape == (3, 3)
    assert model.beta_.shape[1] == 2


def test_nf_rvfl_regressor_single_output_is_one_dimensional():
    X = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y = X[:, 0] ** 2
    model = NeuroFuzzyRVFLRegressor(
        n_fuzzy_nodes=2,
        n_hidden_nodes=3,
        cluster_method="fuzzy_cmeans",
        random_state=11,
    ).fit(X, y)

    assert model.predict(X).shape == (8,)


def test_nf_rvfl_rejects_invalid_parameters_and_predict_before_fit():
    with pytest.raises(ValueError, match="n_hidden_nodes"):
        NeuroFuzzyRVFLClassifier(n_hidden_nodes=0).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="cluster_method"):
        NeuroFuzzyRVFLClassifier(cluster_method="unknown").fit([[0.0], [1.0]], [0, 1])
    with pytest.raises((ValueError, AttributeError)):
        NeuroFuzzyRVFLClassifier().predict([[0.0]])


def test_nf_rvfl_fuzzy_cmeans_handles_a_constant_feature():
    X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]])
    model = NeuroFuzzyRVFLClassifier(
        n_fuzzy_nodes=2,
        n_hidden_nodes=4,
        cmeans_iterations=10,
        random_state=5,
    ).fit(X, [0, 0, 1, 1])

    assert np.isfinite(model.predict_proba(X)).all()
