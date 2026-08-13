
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


def test_agreement_fuzzy():
    import fylearn.fuzzylogic as fl

    A = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    B = np.array([[0.9, 0.8, 0.7], [0.8, 0.7, 0.6]])
    a, d = rafpc.agreement_fuzzy(fl.mean, A, B)
    assert d.shape == (3,)
    assert 0.0 <= a <= 1.0
    # identical samples have perfect agreement
    a2, d2 = rafpc.agreement_fuzzy(fl.mean, A, A)
    assert np.allclose(a2, 1.0)


def test_fuzzify_mean():
    A = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    p, R, mus = rafpc.fuzzify_mean(A)
    assert p == 3
    assert R.shape == (3, 6)
    assert len(mus) == 6


def test_fuzzify_partitions():
    A = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    fuzzify = rafpc.fuzzify_partitions(5)
    p, R, mus = fuzzify(A)
    assert p == 5
    assert R.shape == (3, 10)
    assert len(mus) == 2
    assert len(mus[0]) == 5


def test_factories():
    t = rafpc.triangular_factory(0.0, 0.5, 1.0)
    assert isinstance(t, rafpc.fl.TriangularSet)
    p = rafpc.pi_factory(0.0, 0.5, 1.0)
    assert isinstance(p, rafpc.fl.PiSet)


def test_build_memberships():
    X = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
    mus = rafpc.build_memberships(X, rafpc.triangular_factory)
    assert len(mus) == 2
    assert all(m[0] == i for i, m in enumerate(mus))


def test_agreement_pruning():
    rs = np.random.RandomState(0)
    X = np.array([[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    proto = rafpc.build_memberships(X, rafpc.triangular_factory)
    # prune down to 2 features
    pruned = rafpc.agreement_pruning(X, proto, 2, rs)
    assert len(pruned) == 2
    assert len(rafpc.agreement_pruning(X, pruned, 2, rs)) == 2


def test_build_for_class():
    rs = np.random.RandomState(1)
    X = np.array([[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    proto = rafpc.build_for_class(X, 100, 2, rs, rafpc.triangular_factory)
    assert len(proto) == 2


def test_classifier_pruning_path():
    # n_features < n columns triggers the pruning path
    l = rafpc.RandomAgreementFuzzyPatternClassifier(n_protos=2, n_features=2, random_state=0)
    X = np.array(
        [
            [0.10, 0.20, 0.40],
            [0.15, 0.18, 0.43],
            [0.20, 0.40, 0.80],
            [0.25, 0.42, 0.78],
        ]
    )
    y = np.array([0, 0, 1, 1])
    l.fit(X, y)
    y_pred = l.predict(X)
    assert len(y_pred) == 4


def test_predict_before_fit_raises():
    import pytest

    l = rafpc.RandomAgreementFuzzyPatternClassifier()
    with pytest.raises(Exception) as e:
        l.predict(np.array([[0.1, 0.2]]))
    assert "fit" in str(e.value)


def test_nan_classes_rejected():
    import pytest

    l = rafpc.RandomAgreementFuzzyPatternClassifier()
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    y = np.array([0.0, np.nan])
    with pytest.raises(Exception):
        l.fit(X, y)


def test_n_features_clamped():
    l = rafpc.RandomAgreementFuzzyPatternClassifier(n_features=100, n_protos=1, random_state=0)
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    y = np.array([0, 1])
    l.fit(X, y)
    assert l.n_features == 2
