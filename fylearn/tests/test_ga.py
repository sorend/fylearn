
import numpy as np
from sklearn.utils.testing import assert_true
from fylearn.ga import GeneticAlgorithm

def test_ga_variance():

    # fitness function is the variance (means, prefer with small variance)
    ff = lambda x: np.var(x)
    # create instance
    ga = GeneticAlgorithm(ff, 10, 1000, keep_parents=10, p_mutation=0.1)

    for i in range(100):
        ga.next()

    assert_true(0.01 > ga.best(1)[1])
