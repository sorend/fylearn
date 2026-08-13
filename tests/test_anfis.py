import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fylearn.anfis import AnfisClassifier


def test_anfis_iris():
    # Load data
    data = load_iris()
    X = data.data
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize ANFIS
    anfis = AnfisClassifier(n_rules=3, optimizer_iterations=20, optimizer_pop_size=20, random_state=42)

    # Train
    anfis.fit(X_train, y_train)

    # Predict
    y_pred = anfis.predict(X_test)

    # Check accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"ANFIS Accuracy on Iris: {acc:.4f}")

    # Basic assertion to ensure it learned something better than random guessing
    # Iris has 3 classes, random guess ~0.33
    assert acc > 0.5, f"Accuracy too low: {acc}"

    # Check predict_proba
    probas = anfis.predict_proba(X_test)
    assert probas.shape == (len(X_test), 3)
    # Check row sums
    assert np.allclose(np.sum(probas, axis=1), 1.0)


if __name__ == "__main__":
    test_anfis_iris()


def test_anfis_predict_before_fit():
    import pytest

    anfis = AnfisClassifier()
    with pytest.raises(Exception) as e:
        anfis.predict(np.zeros((2, 4)))
    assert "fit" in str(e.value)


def test_anfis_proba_before_fit():
    import pytest

    anfis = AnfisClassifier()
    with pytest.raises(Exception):
        anfis.predict_proba(np.zeros((2, 4)))


def test_anfis_antecedent_bounds():
    # regression: bounds were previously tiled in a layout inconsistent with
    # the (n_rules, n_features, 3) antecedent layout
    X = np.array(
        [
            [0.0, 10.0],
            [0.5, 20.0],
            [1.0, 30.0],
            [0.0, 15.0],
            [1.0, 25.0],
        ]
    )
    y = np.array([0, 1, 0, 1, 0])

    anfis = AnfisClassifier(n_rules=2, optimizer_iterations=10, optimizer_pop_size=20, random_state=7)
    anfis.fit(X, y)

    # antecedent shape: (n_rules, n_features, 3)
    assert anfis.antecedents_.shape == (2, 2, 3)
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    range_vals = max_vals - min_vals
    lower = min_vals - 0.1 * range_vals
    upper = max_vals + 0.1 * range_vals
    for f in range(2):
        assert np.all(anfis.antecedents_[:, f, :] >= lower[f] - 1e-6)
        assert np.all(anfis.antecedents_[:, f, :] <= upper[f] + 1e-6)
        # each feature's (a, b, c) must be sorted
        assert np.all(np.diff(anfis.antecedents_[:, f, :], axis=1) >= 0)


def test_anfis_small_binary():
    X = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.15],
            [0.1, 0.2],
            [0.9, 0.9],
            [0.8, 0.85],
            [0.9, 0.8],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    anfis = AnfisClassifier(n_rules=3, optimizer_iterations=30, optimizer_pop_size=30, random_state=3)
    anfis.fit(X, y)
    y_pred = anfis.predict(X)
    assert np.mean(y_pred == y) > 0.8
    assert len(anfis.history_) == 30


def test_anfis_t_factory():
    from fylearn.anfis import t_factory
    from fylearn.fuzzylogic import TriangularSet

    t = t_factory(0.5, 0.2)
    assert isinstance(t, TriangularSet)
    assert np.isclose(t.a, 0.3)
    assert np.isclose(t.b, 0.5)
    assert np.isclose(t.c, 0.7)
