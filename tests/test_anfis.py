from fylearn.anfis import AnfisClassifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test_anfis_iris():
    # Load data
    data = load_iris()
    X = data.data
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize ANFIS
    anfis = AnfisClassifier(n_rules=3, optimizer_iterations=20, optimizer_pop_size=20)

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
