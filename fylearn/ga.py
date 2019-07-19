# -*- coding: utf-8 -*-
"""
Genetic algorithm implementation.

The main classes are:

- PointwiseCrossover: Allows to restrict crossover to a selected number of crossover-points.

- UniformCrossover: Performs crossover uniformly for each gene.

Has two selection methods implemented:

- Tournament selection (selects top from a randomly selected tournament)
- Top-n selection (selects top n)

"""
import numpy as np
from sklearn.utils import check_random_state

#
# Authors: Søren A. Davidsen <sorend@cs.svuni.in>
#


def tournament_selection(tournament_size=10):
    def tournament_sel(rs, P, f):
        participants = rs.choice(len(f), size=tournament_size)  # select the tournament participants
        winners = np.argsort(f[participants])  # sort them by fitness
        return participants[winners[0]], participants[winners[1]]  # take two with min fitness

    return tournament_sel

def top_n_selection(n=25):
    def top_n_sel(rs, P, f):
        top_n = np.argsort(f)[:n]  # select top-n fitness.
        parents = rs.choice(top_n, size=2)  # pick two randomly
        return parents[0], parents[1]

    return top_n_sel

def helper_n_generations(ga, n=100):
    for i in range(n):
        ga.next()
    return ga

class UniformCrossover:
    """Implements a uniform crossover, with a specific random probability of getting genes
    from each parent."""

    def __init__(self, p1_proba=0.5):
        """Constructs the crossover operation.

        Parameters:
        -----------

        p1_proba : probability of selecting parent1 (first parent given for crossover operation)
        """
        self.p1_proba = p1_proba

    def __call__(self, P1, P2, random_state):
        random_state = check_random_state(random_state)
        C = np.array(P1)  # clone p1
        R = random_state.random_sample(C.shape) > self.p1_proba  # create filter
        C[R] = np.array(P2, copy=False)[R]  # mixin P2 values
        return C

class PointwiseCrossover:
    """Implements a pointwise crossover operation, meaning crossover operation can only occur
    at predefined locations in the chromosome.
    """
    def __init__(self, crossover_locations, n_crossovers=1):
        """Construct the crossover operation

        Parameters:
        -----------

        crossover_locations : List of locations in the chromosome where crossover can occur.

        n_crossovers : Number of crossovers.
        """
        self.crossover_locations = np.array(crossover_locations, copy=False)
        self.n_crossovers = n_crossovers

    def __call__(self, A, B, random_state):
        random_state = check_random_state(random_state)
        A, B = np.array(A, copy=False), np.array(B, copy=False)
        is_1d = len(A.shape) == 1
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        C = np.zeros(A.shape)

        def pick(a, b, i):
            r = a if i[2] == 0 else b
            return r[i[0]:i[1]].tolist()

        start = [0]
        end = [A.shape[1]]

        for idx, a in enumerate(A):
            b = B[idx]
            selected = np.sort(random_state.choice(self.crossover_locations, self.n_crossovers))
            # use python to merge
            selected = start + selected.tolist() + end
            index = zip(selected, selected[1:], [0, 1] * len(selected))
            merged = np.array([ item for i in index for item in pick(a, b, i) ])
            # add merged child
            C[idx, :] = merged

        if is_1d:
            return C.ravel()
        else:
            return C

def helper_min_fitness_decrease(ga, epsilon=0.001, top_n=10):
    last_fitness = None
    while True:
        # next
        ga.next()
        # get top_n
        chromosomes, fitnesses = ga.best(top_n)
        # mean fitness
        new_fitness = np.mean(fitnesses)
        if last_fitness is not None:
            d_fitness = last_fitness - new_fitness
            if d_fitness < epsilon:
                break
        last_fitness = new_fitness
    return ga

def helper_fitness(chromosome_fitness_function):
    """
    Helper function, will evaluate chromosome_fitness_function for each chromosome
    and return the result. Can be used to wrap fitness functions that evaluate
    each chromosome at a time instead of the whole population.
    """
    def fitness_function(population):
        return np.apply_along_axis(chromosome_fitness_function, 1, population)
    return fitness_function

class BaseGeneticAlgorithm(object):

    def __init__(self, fitness_function,
                 selection_function=tournament_selection(),
                 n_genes=None, n_chromosomes=None,
                 elitism=0, p_mutation=0.1,
                 random_state=None,
                 population=None,
                 crossover_function=UniformCrossover()):
        """
        Initializes the genetic algorithm.

        Parameters:
        -----------

        fitness_function : the function to measure fitness (smaller is more "fit")

        n_genes : number of genes (only needed if population is not specified)

        n_chromosomes : number of chromosomes (only needed if population is not specified)

        elitism : number of parents to keep between each generation

        p_mutation : the probability for mutation (applies to each gene)

        random_state : the state for the randomization to use

        population : Initial population, use this or use n_genes and n_chromosomes

        crossover_function : the function to perform crossover between two parents

        """
        self.fitness_function = fitness_function
        self.selection_function = selection_function
        self.elitism = elitism
        self.p_mutation = p_mutation
        self.random_state = check_random_state(random_state)

        # init population
        if population is None:
            self.population_ = self.initialize_population(n_chromosomes, n_genes)
            self.n_genes = n_genes
            self.n_chromosomes = n_chromosomes
        else:
            self.population_, = population
            self.n_genes = self.population_.shape[1]
            self.n_chromosomes = self.population_.shape[0]
        # crossover
        self.crossover_function = crossover_function
        # init fitness
        self.fitness_ = self.fitness_function(self.population_)

    def initialize_population(self, n_chromosomes, n_genes):
        raise Exception("initialize_population not implemented")

    def mutate(self, chromosomes, mutation_idx):
        raise Exception("mutate not implemented")

    def next(self):
        # create new population
        new_population = np.array(self.population_)

        # if we have elitism, sort so we have the top the most fit.
        if self.elitism > 0:
            new_population = new_population[np.argsort(self.fitness_)]

        fathers = np.zeros(self.population_.shape)
        mothers = np.zeros(self.population_.shape)

        # selection, generate sets of parents
        for idx in range(self.elitism, self.n_chromosomes):
            father_idx, mother_idx = self.selection_function(self.random_state, self.population_, self.fitness_)
            fathers[idx], mothers[idx] = self.population_[father_idx], self.population_[mother_idx]

        # generate new children
        new_population[self.elitism:] = self.crossover_function(fathers[self.elitism:],
                                                                mothers[self.elitism:],
                                                                self.random_state)

        # mutate
        mutation_idx = self.random_state.random_sample(new_population[self.elitism:].shape) < self.p_mutation
        new_population[self.elitism:] = self.mutate(new_population[self.elitism:],
                                                    mutation_idx)

        # update pop and fitness
        self.population_ = new_population
        self.fitness_ = self.fitness_function(self.population_)

    def best(self, n_best=1):
        f_sorted = np.argsort(self.fitness_)
        p_sorted = self.population_[f_sorted]
        return p_sorted[:n_best], self.fitness_[f_sorted][:n_best]

class GeneticAlgorithm(BaseGeneticAlgorithm):
    """
    Continuous genetic algorithm is a genetic algorithm where the gene values
    are chosen from a continous domain. This means the population is randomly
    initialized to [-1.0, 1.0] and mutation modifies the gene value in the range [-1.0, 1.0].
    """
    def __init__(self, scaling=1.0, *args, **kwargs):
        self.scaling = scaling
        super(GeneticAlgorithm, self).__init__(*args, **kwargs)

    def initialize_population(self, n_chromosomes, n_genes):
        return self.random_state.rand(n_chromosomes, n_genes) * self.scaling

    def mutate(self, chromosomes, mutation_idx):
        mutations = (self.random_state.rand(np.sum(mutation_idx)) - 0.5) * self.scaling
        chromosomes[mutation_idx] += mutations
        return chromosomes

class UnitIntervalGeneticAlgorithm(GeneticAlgorithm):
    """
    Genetic algorithm where gene values are chosen from the unit interval [0, 1]. Mutation
    randomly selects a new value in this interval.
    """
    def mutate(self, chromosomes, mutation_idx):
        mutations = self.random_state.rand(np.sum(mutation_idx)) * self.scaling
        chromosomes[mutation_idx] = mutations
        return chromosomes

class DiscreteGeneticAlgorithm(GeneticAlgorithm):
    """
    Discrete genetic algorithm is a genetic algorithm where the gene values are
    chosen from a discrete domain. In this context, mutation means to randomly pick another value
    from this domain.
    """

    def __init__(self, ranges=None, *args, **kwargs):
        """
        Initializes the genetic algorithm.

        Parameters:
        -----------

        ranges : An tuple of tuples describing the values possible for each gene.
                 (ranges[gene_index] are the selectable values for the gene)
        """
        self.ranges = ranges
        super(DiscreteGeneticAlgorithm, self).__init__(*args, **kwargs)

    def initialize_population(self, n_chromosomes, n_genes):
        P = np.zeros((n_chromosomes, n_genes))
        for i in range(n_genes):
            P[:, i] = self.random_state.choice(self.ranges[i], P.shape[0])
        return P

    def mutate(self, chromosomes, mutation_idx):
        for i in range(self.n_genes):
            midx_i = mutation_idx[:, i]
            chromosomes[midx_i, i] = self.random_state.choice(self.ranges[i], np.sum(midx_i))
        return chromosomes
