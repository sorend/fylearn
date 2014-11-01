
import numpy as np
from sklearn.utils.testing import assert_equal

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
    assert_equal(3, i)

    # check we really got preorder    
    t = list(fpt._tree_iterator(root))
    assert_equal(3, len(t))
    assert_equal(root, t[0])
    assert_equal(l1, t[1])
    assert_equal(l2, t[2])
    
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

    assert_equal([0], l.predict([[0.9, 1.7, 4.5]]))

def test_classifier_single():

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

    assert_equal(0, l.predict([0.9, 1.7, 4.5]))
        
