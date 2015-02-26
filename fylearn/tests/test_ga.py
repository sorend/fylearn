
import numpy as np
from sklearn.utils.testing import assert_true
from fylearn.ga import GeneticAlgorithm, tournament_selection, DiscreteGeneticAlgorithm

def test_discrete():

    ff = lambda x: np.var(x)

    ranges = (
        range(10),
        range(20, 30),
        range(40, 50),
        range(2),
    )
        
    ga = DiscreteGeneticAlgorithm(fitness_function=ff, n_genes=4, n_chromosomes=100, p_mutation=0.1,
                                  ranges=ranges)


    for i in range(10):
        print ga.best()
        ga.next()

    assert_true(ga.best()[1] <= np.var([9, 20, 40, 1])) # assume we find the solution


def test_tournament_selection():

    X = np.random.rand(5, 3)
    f = np.array([0.6, 0.2, 0.3, 0.4, 0.1])

    sel = tournament_selection(3)

    rs = np.random.RandomState()

    p, q = sel(rs, X, f)

    print "p", p, "q", q, "f[p]", f[p], "f[q]", f[q]

    assert_true(f[p] <= f[q])


def test_sreedevi():

    # use ga to solve a + 2b + 3c + 4d = 30

    ff = lambda x: 1.0 / (x[0] + (2 * x[1]) + (3 * x[2]) + (4 * x[3]) - 30)

    ga = GeneticAlgorithm(fitness_function=ff, n_genes=4, n_chromosomes=100, p_mutation=0.1)

    for i in range(100):
        ga.next()

    chromosomes, fitness = ga.best(1)

    # take first one
    c = chromosomes[0]

    print "c", c
    print "f(c)", c[0] + (2 * c[1]) + (3 * c[2]) + (4 * c[3])

    assert_true((c[0] + (2 * c[1]) + (3 * c[2]) + (4 * c[3]) - 30.0) < 0.1)


def test_ga_variance():

    # fitness function is the variance (means, prefer with small variance)
    ff = lambda x: np.var(x)
    # create instance
    ga = GeneticAlgorithm(fitness_function=ff, n_genes=10, n_chromosomes=1000, elitism=10, p_mutation=0.1)

    for i in range(100):
        print "next generation", i
        ga.next()

    assert_true(0.01 > ga.best(1)[1])

