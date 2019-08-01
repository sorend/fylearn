from __future__ import print_function

import numpy as np
# from sklearn.utils.testing import assert_equal, assert_true

import fylearn.rafpc as rafpc

def test_agreement_hamming():

    X = np.array([
        [0.1,  0.3,  0.4,  0.1, 0.3, 0.4],
        [0.09, 0.28, 0.45, 0.4, 0.3, 0.1]
    ])

    e = rafpc.agreement_hamming(3, X, 0, 1)

    print("e", e)

    assert e[0] > 0.9
    assert e[1] < 0.9

def test_classifier():

    l = rafpc.RandomAgreementFuzzyPatternClassifier(n_protos=1, n_features=2)

    X = np.array([
        [0.10, 0.20, 0.40],
        [0.15, 0.18, 0.43],
        [0.20, 0.40, 0.80],
        [0.25, 0.42, 0.78]
    ])

    y = np.array([
        0,
        0,
        1,
        1
    ])

    l.fit(X, y)

    assert [0] == l.predict([[0.9, 1.7, 4.5]])

# def test_classifier_iris():

#     import os
#     csv_file = os.path.join(os.path.dirname(__file__), "iris.csv")
#     data = np.genfromtxt(csv_file, dtype=float, delimiter=',', names=True)

#     X = np.array([data["sepallength"], data["sepalwidth"], data["petallength"], data["petalwidth"]]).T
#     y = data["class"]

#     from sklearn.preprocessing import MinMaxScaler
#     X = MinMaxScaler().fit_transform(X)

#     l = rafpc.RandomAgreementFuzzyPatternClassifier(n_protos=10, random_state=0, n_features=3)

#     from sklearn import cross_validation

#     scores = cross_validation.cross_val_score(l, X, y, cv=10)
#     mean = np.mean(scores)

#     print "mean", mean

#     assert_true(0.92 < mean and mean < 0.94)
