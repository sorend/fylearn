
import numpy as np
from sklearn.utils.testing import assert_equal, assert_true

import fylearn.fuzzylogic as fl
    
def test_owa():

    X = np.array([1.0, 1.0, 1.0, 1.0])
    w = [0.5, 0.3, 0.2]

    owa = fl.owa(w)

    r = owa(X)

    assert_equal(1.0, r)

    owa = fl.owa(0.5, 0.3, 0.2)
    r2 = owa(X)

    assert_equal(r, r2)

def test_owa_matrix():

    X = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    w = [0.5, 0.3, 0.2]

    owa = fl.owa(w)

    r = owa(X)

    assert_equal(1.0, r[0])
    assert_equal(0.5, r[1])

