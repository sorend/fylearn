from __future__ import print_function

import numpy as np

import fylearn.nfpc as nfpc
import fylearn.fuzzylogic as fl
from sklearn import datasets

def t_factory(**k):
    return fl.TriangularSet(k["min"], k["mean"], k["max"])

def test_simple_fpc():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    c = nfpc.FuzzyPatternClassifier()
    c.fit(X, y)
    mu = nfpc.predict_protos(X, c.protos_, c.aggregation_)
    pro = c.predict_proba(X)
    pro_sum = pro.sum(axis=1)
    assert np.all(pro_sum > 0.)

# def test_build_shrinking():

#     X = np.random.rand(100, 3)
#     X[:, 1] = X[:, 1] * 10
#     X[:, 2] = X[:, 2] * 50

#     impl = nfpc.IterativeShrinking(alpha_cut=0.1, iterations=2)
#     s = impl(X, t_factory)

#     assert len(s) == 3
#     assert s[0].b - 0.5 < 0.05
#     assert s[1].b - 5 < 1
#     assert s[2].b - 25 < 3


# def test_build_shrinking_data():

#     import os
#     csv_file = os.path.join(os.path.dirname(__file__), "iris.csv")
#     data = np.genfromtxt(csv_file, dtype=float, delimiter=',', names=True)

#     X = np.array([data["sepallength"], data["sepalwidth"], data["petallength"], data["petalwidth"]]).T
#     y = data["class"]

#     from sklearn.preprocessing import MinMaxScaler
#     X = MinMaxScaler().fit_transform(X)

#     l = nfpc.ShrinkingFuzzyPatternClassifier(shrinking=nfpc.IterativeShrinking(iterations=2, alpha_cut=0.05),
#                                              membership_factory=t_factory)

#     from sklearn import cross_validation

#     scores = cross_validation.cross_val_score(l, X, y, cv=10)
#     mean = np.mean(scores)

#     print "mean", mean

#     assert 0.80 < mean


def test_build_meowa_factory():

    import os
    csv_file = os.path.join(os.path.dirname(__file__), "iris.csv")
    data = np.genfromtxt(csv_file, dtype=float, delimiter=',', names=True)

    X = np.array([data["sepallength"], data["sepalwidth"], data["petallength"], data["petalwidth"]]).T
    y = data["class"]

    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)

    l = nfpc.FuzzyPatternClassifier(membership_factory=t_factory,
                                    aggregation_factory=nfpc.MEOWAFactory())

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(l, X, y, cv=10)
    mean = np.mean(scores)

    assert 0.80 < mean


def test_build_ps_owa_factory():

    import os
    csv_file = os.path.join(os.path.dirname(__file__), "iris.csv")
    data = np.genfromtxt(csv_file, dtype=float, delimiter=',', names=True)

    X = np.array([data["sepallength"], data["sepalwidth"], data["petallength"], data["petalwidth"]]).T
    y = data["class"]

    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)

    l = nfpc.FuzzyPatternClassifier(
        membership_factory=t_factory,
        aggregation_factory=nfpc.GAOWAFactory(optimizer=nfpc.ps_owa_optimizer())
    )

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(l, X, y, cv=10)
    mean = np.mean(scores)

    print("mean", mean)

    assert 0.92 < mean
