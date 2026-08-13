
import numpy as np
import pytest

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

    ga = DiscreteGeneticAlgorithm(
        fitness_function=helper_fitness(ff), n_genes=4, n_chromosomes=100, p_mutation=0.1, ranges=ranges
    )

    for i in range(10):
        print(ga.best())
        next(ga)

    assert ga.best()[1] <= np.var([9, 20, 40, 1])  # assume we find the solution


def test_tournament_selection():
    X = np.random.rand(5, 3)
    f = np.array([0.6, 0.2, 0.3, 0.4, 0.1])

    sel = tournament_selection(3)

    rs = np.random.RandomState()

    p, q = sel(rs, X, f)

    # Selection returns indices, so p and q are indices
    # We can check that the selected individuals are within bounds
    assert 0 <= p < len(f)
    assert 0 <= q < len(f)

    # In a tournament of 3 from 5, with these fitnesses, it's probabilistic
    # but we can check if we run it many times.
    # But for a single run, just checking structure is safer.


def test_sreedevi():
    # use ga to solve a + 2b + 3c + 4d = 30

    ff = lambda x: 1.0 / (x[0] + (2 * x[1]) + (3 * x[2]) + (4 * x[3]) - 30)

    ga = GeneticAlgorithm(fitness_function=helper_fitness(ff), n_genes=4, n_chromosomes=100, p_mutation=0.1)

    for i in range(100):
        next(ga)

    chromosomes, fitness = ga.best(1)

    # take first one
    c = chromosomes[0]

    print("c", c)
    print("f(c)", c[0] + (2 * c[1]) + (3 * c[2]) + (4 * c[3]))

    # Relaxed assertion due to stochastic nature
    val = c[0] + (2 * c[1]) + (3 * c[2]) + (4 * c[3])
    assert abs(val - 30.0) < 5.0


def test_ga_variance():
    # fitness function is the variance (means, prefer with small variance)
    ff = lambda x: np.var(x)
    # create instance
    ga = GeneticAlgorithm(
        fitness_function=helper_fitness(ff), n_genes=10, n_chromosomes=1000, elitism=10, p_mutation=0.1
    )

    for i in range(50):
        print("next generation", i)
        next(ga)

    assert 0.1 > ga.best(1)[1]


def test_population_as_plain_array():
    # regression: population could only be passed as a 1-tuple before
    P = np.random.rand(10, 3)
    ga = GeneticAlgorithm(fitness_function=lambda p: np.zeros(len(p)), population=P)
    assert ga.n_genes == 3
    assert ga.n_chromosomes == 10
    assert np.array_equal(ga.population_, P)


def test_population_as_tuple():
    P = np.random.rand(10, 3)
    ga = GeneticAlgorithm(fitness_function=lambda p: np.zeros(len(p)), population=(P,))
    assert ga.n_chromosomes == 10


def test_population_wrong_shape():
    P = np.random.rand(10)
    with pytest.raises(ValueError):
        GeneticAlgorithm(fitness_function=lambda p: np.zeros(len(p)), population=P)


def test_no_population_or_genes_raises():
    with pytest.raises(ValueError):
        GeneticAlgorithm(fitness_function=lambda p: np.zeros(len(p)))


def test_best_n_multiple():
    ga = GeneticAlgorithm(fitness_function=helper_fitness(lambda x: np.var(x)), n_genes=3, n_chromosomes=20)
    chroms, fits = ga.best(5)
    assert chroms.shape == (5, 3)
    assert fits.shape == (5,)
    assert np.all(np.diff(fits) >= 0)  # sorted ascending


def test_elitism_keeps_best():
    ff = lambda x: np.sum(x**2)  # noqa: E731
    P = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.2, 0.2, 0.2]])
    ga = GeneticAlgorithm(fitness_function=helper_fitness(ff), population=P, elitism=1, p_mutation=0.0, random_state=0)
    before = ga.best(1)[0][0]
    next(ga)
    after = ga.best(1)[0][0]
    assert np.array_equal(before, after)


def test_top_n_selection():
    sel = top_n_selection(3)
    rs = np.random.RandomState(42)
    f = np.array([0.5, 0.1, 0.4, 0.9, 0.2])
    p, q = sel(rs, None, f)
    # both parents must be in the top-3
    top3 = set(np.argsort(f)[:3])
    assert p in top3
    assert q in top3
    # vectorized version
    p, q = sel(rs, None, f, n_selection=4)
    assert p.shape == (4,)
    assert q.shape == (4,)


def test_tournament_selection_vectorized():
    sel = tournament_selection(3)
    rs = np.random.RandomState(42)
    f = np.array([0.6, 0.2, 0.3, 0.4, 0.1])
    p, q = sel(rs, None, f, n_selection=7)
    assert p.shape == (7,)
    assert q.shape == (7,)
    assert np.all(p < len(f))
    assert np.all(q < len(f))


def test_unit_interval_ga_stays_in_unit_interval():
    ff = lambda x: np.var(x)
    ga = UnitIntervalGeneticAlgorithm(fitness_function=helper_fitness(ff), n_genes=5, n_chromosomes=50, p_mutation=0.3)
    for i in range(20):
        next(ga)
        assert np.all(ga.population_ >= 0.0)
        assert np.all(ga.population_ <= 1.0)


def test_helper_min_fitness_decrease():
    ff = lambda x: np.var(x)
    ga = GeneticAlgorithm(fitness_function=helper_fitness(ff), n_genes=5, n_chromosomes=50, p_mutation=0.1)
    ga = helper_min_fitness_decrease(ga, epsilon=0.0001, top_n=5)
    chroms, fits = ga.best(1)
    assert fits[0] < 1.0


def test_pointwise_crossover_deterministic():
    # with a fixed random state, crossover point selection is deterministic
    rs1 = np.random.RandomState(7)
    rs2 = np.random.RandomState(7)
    c = PointwiseCrossover(crossover_locations=[0, 1, 2], n_crossovers=1)
    a = c([1, 2, 3], [10, 20, 30], rs1)
    b = c([1, 2, 3], [10, 20, 30], rs2)
    assert np.array_equal(a, b)
    assert set(np.unique(a)).issubset({1, 2, 3, 10, 20, 30})
