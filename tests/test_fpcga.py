from __future__ import print_function

import numpy as np

import fylearn.fpcga as fpcga

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

    import os
    csv_file = os.path.join(os.path.dirname(__file__), "iris.csv")
    data = np.genfromtxt(csv_file, dtype=float, delimiter=',', names=True)

    X = np.array([data["sepallength"], data["sepalwidth"], data["petallength"], data["petalwidth"]]).T
    y = data["class"]

    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)

    l = fpcga.FuzzyPatternClassifierGA(iterations=100)

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(l, X, y, cv=10)
    mean = np.mean(scores)

    print("mean", mean)

    assert 0.90 < mean
