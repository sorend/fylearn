from __future__ import print_function

import numpy as np
from fylearn.ga import *


def test_uniform_crossover():
    
    rs = np.random.RandomState(0)
    
    c = UniformCrossover()

    a = c([1, 2, 3], [1, 2, 3], rs)

    assert len(a.shape) == 1
    assert a.shape[0] == 3
    assert 1 == a[0]
    assert 2 == a[1]
    assert 3 == a[2]

    b = c([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], rs)

    assert len(b.shape) == 2
    assert b.shape[0] == 2
    assert b.shape[1] == 3


def test_pointwise_crossover():

    rs = np.random.RandomState(0)

    p1 = range(0, 10, 1)
    p2 = range(0, 100, 10)
    
    c = PointwiseCrossover(crossover_locations=[1, 3, 5, 7, 8], n_crossovers=2)

    a = c(p1, p2, rs)

    assert len(a.shape) == 1
    assert a.shape[0] == 10

    b = c([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], rs)

    assert len(b.shape) == 2
    assert b.shape[0] == 2
    assert b.shape[1] == 3


def test_discrete():

    ff = lambda x: np.var(x)

    ranges = (
        range(10),
        range(20, 30),
        range(40, 50),
        range(2),
    )

    ga = DiscreteGeneticAlgorithm(fitness_function=helper_fitness(ff),
                                  n_genes=4, n_chromosomes=100, p_mutation=0.1,
                                  ranges=ranges)

    for i in range(10):
        print(ga.best())
        ga.next()

    assert ga.best()[1] <= np.var([9, 20, 40, 1])  # assume we find the solution


def test_tournament_selection():

    X = np.random.rand(5, 3)
    f = np.array([0.6, 0.2, 0.3, 0.4, 0.1])

    sel = tournament_selection(3)

    rs = np.random.RandomState()

    p, q = sel(rs, X, f)

    print("p", p, "q", q, "f[p]", f[p], "f[q]", f[q])

    assert f[p] <= f[q]


def test_sreedevi():

    # use ga to solve a + 2b + 3c + 4d = 30

    ff = lambda x: 1.0 / (x[0] + (2 * x[1]) + (3 * x[2]) + (4 * x[3]) - 30)

    ga = GeneticAlgorithm(fitness_function=helper_fitness(ff), n_genes=4, n_chromosomes=100, p_mutation=0.1)

    for i in range(100):
        ga.next()

    chromosomes, fitness = ga.best(1)

    # take first one
    c = chromosomes[0]

    print("c", c)
    print("f(c)", c[0] + (2 * c[1]) + (3 * c[2]) + (4 * c[3]))

    assert (c[0] + (2 * c[1]) + (3 * c[2]) + (4 * c[3]) - 30.0) < 0.1


def test_ga_variance():

    # fitness function is the variance (means, prefer with small variance)
    ff = lambda x: np.var(x)
    # create instance
    ga = GeneticAlgorithm(fitness_function=helper_fitness(ff), n_genes=10,
                          n_chromosomes=1000, elitism=10, p_mutation=0.1)

    for i in range(100):
        print("next generation", i)
        ga.next()

    assert 0.01 > ga.best(1)[1]
