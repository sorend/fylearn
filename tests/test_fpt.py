
import numpy as np
import pytest
from sklearn.datasets import load_iris

import fylearn.fpt as fpt


def test_tree_iterator():
    """ tests tree iterator """

    def f(x):
        return 1.0

    l1, l2 = fpt.Leaf(0, "l1", f), fpt.Leaf(1, "l2", f)
    root = fpt.Inner(max, [l1, l2])

    i = 0
    for x in fpt._tree_iterator(root):
        i += 1

    # check all have been iterated
    assert 3 == i

    # check we really got preorder
    t = list(fpt._tree_iterator(root))
    assert 3 == len(t)
    assert root == t[0]
    assert l1 == t[1]
    assert l2 == t[2]

def test_classifier():

    l = fpt.FuzzyPatternTreeClassifier()

    X = np.array([
        [1.0, 2.0, 4.0],
        [2.0, 4.0, 8.0]
    ])

    y = np.array([
        0,
        1
    ])

    l.fit(X, y)

    assert [0] == l.predict([[0.9, 1.7, 4.5]])

def test_classifier_topdown():

    l = fpt.FuzzyPatternTreeTopDownClassifier()

    X = np.array([
        [1.0, 2.0, 4.0],
        [2.0, 4.0, 8.0]
    ])

    y = np.array([
        0,
        1
    ])

    l.fit(X, y)

    assert [0] == l.predict([[0.9, 1.7, 4.5]])

def test_classifier_iris():

    iris = load_iris()

    X = iris.data
    y = iris.target

    l = fpt.FuzzyPatternTreeClassifier()
    l.fit(X, y)
    score = l.score(X, y)
    print("score", score)

    assert 0.97 == pytest.approx(score, 0.01)


def test_leaf_call_and_repr():
    t = fpt.fl.TriangularSet(0.0, 0.5, 1.0)
    leaf = fpt.Leaf(0, "low", t)
    X = np.array([[0.0, 0.5], [0.5, 0.5]])
    y = leaf(X)
    assert y.shape == (2,)
    assert np.isclose(y[0], 0.0)
    assert np.isclose(y[1], 1.0)
    assert repr(leaf) == "Leaf(0_low)"


def test_inner_repr_with_function():
    def f(x):
        return 1.0

    l1, l2 = fpt.Leaf(0, "l1", f), fpt.Leaf(1, "l2", f)
    root = fpt.Inner(fpt.fl.prod, [l1, l2])
    assert "prod" in repr(root)
    assert "Leaf" in repr(root)


def test_inner_repr_with_owa():
    # regression: OWA objects have no __name__ attribute
    t = fpt.fl.TriangularSet(0.0, 0.5, 1.0)
    l1, l2 = fpt.Leaf(0, "l1", t), fpt.Leaf(1, "l2", t)
    root = fpt.Inner(fpt.fl.owa(0.3, 0.7), [l1, l2])
    r = repr(root)
    assert "OWA" in r


def test_inner_call():
    t = fpt.fl.TriangularSet(0.0, 0.5, 1.0)
    l1, l2 = fpt.Leaf(0, "l1", t), fpt.Leaf(1, "l2", t)
    root = fpt.Inner(fpt.fl.prod, [l1, l2])
    X = np.array([[0.5, 0.5]])
    y = root(X)
    assert np.isclose(y[0], 1.0)


def test_tree_contains():
    t = fpt.fl.TriangularSet(0.0, 0.5, 1.0)
    l1, l2 = fpt.Leaf(0, "l1", t), fpt.Leaf(1, "l2", t)
    root = fpt.Inner(fpt.fl.prod, [l1, l2])
    assert fpt._tree_contains(root, l1)
    assert fpt._tree_contains(root, l2)
    assert not fpt._tree_contains(root, fpt.Leaf(2, "l3", t))


def test_tree_leaves():
    t = fpt.fl.TriangularSet(0.0, 0.5, 1.0)
    l1, l2, l3 = fpt.Leaf(0, "l1", t), fpt.Leaf(1, "l2", t), fpt.Leaf(2, "l3", t)
    root = fpt.Inner(fpt.fl.prod, [fpt.Inner(fpt.fl.mean, [l1, l2]), l3])
    leaves = fpt._tree_leaves(root)
    assert len(leaves) == 3


def test_tree_clone_replace_leaf():
    t = fpt.fl.TriangularSet(0.0, 0.5, 1.0)
    l1, l2 = fpt.Leaf(0, "l1", t), fpt.Leaf(1, "l2", t)
    root = fpt.Inner(fpt.fl.prod, [l1, l2])
    new_leaf = fpt.Leaf(2, "l3", t)
    cloned = fpt._tree_clone_replace_leaf(root, l1, new_leaf)
    assert fpt._tree_contains(cloned, new_leaf)
    assert not fpt._tree_contains(cloned, l1)
    assert fpt._tree_contains(cloned, l2)
    # root itself can be replaced
    assert fpt._tree_clone_replace_leaf(root, root, new_leaf) is new_leaf


def test_tree_evaluator_cache():
    calls = []

    class CountingSet:
        def __call__(self, X):
            calls.append(1)
            return np.ones(len(X))

    leaf = fpt.Leaf(0, "c", CountingSet())
    ev = fpt.TreeEvaluator(np.zeros((2, 1)))
    ev.predict(leaf)
    ev.predict(leaf)
    assert len(calls) == 1  # cached


def test_classifier_predict_before_fit():
    l = fpt.FuzzyPatternTreeClassifier()
    with pytest.raises(Exception) as e:
        l.predict([[0.9, 1.7, 4.5]])
    assert "fit" in str(e.value)


def test_classifier_nan_classes_rejected():
    l = fpt.FuzzyPatternTreeClassifier()
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    y = np.array([0.0, 1.0, np.nan])
    with pytest.raises(Exception):
        l.fit(X, y)


def test_classifier_params():
    l = fpt.FuzzyPatternTreeClassifier(max_depth=3, num_candidates=4)
    p = l.get_params()
    assert p["max_depth"] == 3
    assert p["num_candidates"] == 4
    l.set_params(max_depth=2)
    assert l.max_depth == 2


def test_topdown_params():
    l = fpt.FuzzyPatternTreeTopDownClassifier(relative_improvement=0.5)
    p = l.get_params()
    assert p["relative_improvement"] == 0.5
    l.set_params(relative_improvement=0.1)
    assert l.relative_improvement == 0.1


def test_default_fuzzifier():
    leaves = fpt.default_fuzzifier(0, np.array([0.0, 1.0, 2.0]))
    assert len(leaves) == 3
    assert [x.name for x in leaves] == ["low", "med", "hig"]
    assert all(isinstance(x, fpt.Leaf) for x in leaves)


def test_select_slaves():
    X = np.array([[0.1, 0.2], [0.2, 0.4], [0.9, 0.8]])
    class_vector = np.array([1.0, 0.0, 0.0])
    leaves = fpt.default_fuzzifier(0, X[:, 0]) + fpt.default_fuzzifier(1, X[:, 1])
    l = fpt.FuzzyPatternTreeClassifier()
    candidates = fpt._select_candidates(leaves, 3, class_vector, fpt.default_rmse, fpt.TreeEvaluator(X))
    slaves = l.select_slaves(candidates, leaves, class_vector, X)
    assert len(slaves) <= l.num_slaves
    assert all(isinstance(s[1], fpt.Inner) for s in slaves)
