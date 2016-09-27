
from __future__ import print_function

import numpy as np

from fylearn.frr import ModifiedFuzzyPatternClassifier as MFPC

def test_classifier():

    l = MFPC()

    print("l", l.get_params())

    X = np.array([
        [0.1, 0.2, 0.4],
        [0.11, 0.3, 0.5],
        [0.2, 0.4, 0.8],
        [0.18, 0.42, 0.88]
    ])

    y = np.array([
        1,
        1,
        0,
        0
    ])

    l.fit(X, y)

    y_pred = l.predict([[0.0, 0.3, 0.35],
                        [0.1, 0.4, 0.78]])

    print("y_pred", y_pred)

    assert len(y_pred) == 2
    assert y_pred[0] == 1
    assert y_pred[1] == 0
