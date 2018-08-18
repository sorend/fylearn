from __future__ import print_function
import numpy as np

from fylearn import garules

def test_classifier():

    l = garules.EnsembleMultimodalEvolutionaryClassifier(n_iterations=25)

    X = np.array([
        [1, 2, 4],
        [2, 4, 8]
    ])

    y = np.array([
        0,
        1
    ])

    l.fit(X, y)

    print("models", l.models_)

    assert [0] == l.predict([[0.9, 1.7, 4.5]])

    assert [1] == l.predict([[2.1, 3.9, 7.8]])

# def test_compare_diabetes():
#     import os
#     csv_file = os.path.join(os.path.dirname(__file__), "diabetes.csv")
#     data = np.genfromtxt(csv_file, dtype=float, delimiter=',', names=True)

#     X = np.array([data["preg"], data["plas"], data["pres"], data["skin"],
#                   data["insu"], data["mass"], data["pedi"], data["age"]]).T
#     y = data["class"]

#     #f = (data["plas"] > 0.0) & (data["pres"] > 0.0) & (data["skin"] > 0.0) & (data["mass"] > 0.0)

#     #X = X[f]
#     #y = y[f]

#     print "len(X)", len(X)

#     from sklearn.preprocessing import MinMaxScaler
#     X = MinMaxScaler().fit_transform(X)

#     l = garules.EnsembleMultimodalEvolutionaryClassifier(n_iterations=25, n_models=5)

#     from sklearn import cross_validation

#     scores = cross_validation.cross_val_score(l, X, y, cv=10)
#     mean = np.mean(scores)

#     print "mean ensemble", mean

#     l = garules.MultimodalEvolutionaryClassifier(n_iterations=25)
#     scores = cross_validation.cross_val_score(l, X, y, cv=10)
#     mean = np.mean(scores)
#     print "mean normal", mean
#     #assert_true(0.68 < mean)
