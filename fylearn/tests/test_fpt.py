from __future__ import print_function

import numpy as np

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

    import os
    csv_file = os.path.join(os.path.dirname(__file__), "iris.csv")
    data = np.genfromtxt(csv_file, dtype=float, delimiter=',', names=True)

    X = np.array([data["sepallength"], data["sepalwidth"], data["petallength"], data["petalwidth"]]).T
    y = data["class"]

    l = fpt.FuzzyPatternTreeClassifier()
    l.fit(X, y)
    print(l.score(X, y))
