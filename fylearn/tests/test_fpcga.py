
import numpy as np
from sklearn.utils.testing import assert_equal, assert_true

import fylearn.fpcga as fpcga
    
def test_classifier():

    l = fpcga.FuzzyPatternClassifierGA(iterations=100)

    X = np.array([
        [0.1, 0.2, 0.4],
        [0.2, 0.4, 0.8]
    ])

    y = np.array([
        0,
        1
    ])
    
    l.fit(X, y)

    assert_equal([0], l.predict([[0.9, 1.7, 4.5]]))

def test_classifier_single():

    l = fpcga.FuzzyPatternClassifierGA(iterations=100)

    X = np.array([
        [0.1, 0.2, 0.4],
        [0.2, 0.4, 0.8]
    ])

    y = np.array([
        0,
        1
    ])
    
    l.fit(X, y)

    assert_equal(0, l.predict([0.9, 1.7, 4.5]))
        
        
def test_classifier_iris():

    import os
    csv_file = os.path.join(os.path.dirname(__file__), "iris.csv")
    data = np.genfromtxt(csv_file, dtype=float, delimiter=',', names=True)

    X = np.array([data["sepallength"], data["sepalwidth"], data["petallength"], data["petalwidth"]]).T
    y = data["class"]

    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)

    l = fpcga.FuzzyPatternClassifierGA(iterations=100)

    from sklearn import cross_validation

    scores = cross_validation.cross_val_score(l, X, y, cv=10)
    mean = np.mean(scores)

    print "mean", mean

    assert_true(0.90 < mean)
    
