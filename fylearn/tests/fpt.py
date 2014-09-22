
import numpy as np
from sklearn.utils.testing import assert_equal

import fylearn.fpt as fpt

def test_tree_iterator():
    """ tests tree iterator """

    def f(x):
        return 1.0
    
    l1, l2 = fpt.Leaf(0, f), fpt.Leaf(1, f)
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
    
