# -*- coding: utf-8 -*-
"""
Genetic algorithm implementation.


"""
import numpy as np
from sklearn.utils import check_arrays

#
# Authors: Søren A. Davidsen <sorend@cs.svuni.in>
#

def tournament_selection(tournament_size=10):
    def tournament_sel(rs, P, f):
        participants = rs.choice(len(f), size=tournament_size) # select the tournament participants
        winners = np.argsort(f[participants]) # sort them by fitness
        return participants[winners[0]], participants[winners[1]] # take two with min fitness

    return tournament_sel

def top_n_selection(n=25):
    def top_n_sel(rs, P, f):
        top_n = np.argsort(f)[:n] # select top-n fitness.
        parents = rs.choice(top_n, size=2) # pick two randomly
        return parents[0], parents[1]

    return top_n_sel

def helper_n_generations(ga, n=100):
    for i in range(n):
        ga.next()
    return ga

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

class BaseGeneticAlgorithm(object):

    def __init__(self, fitness_function,
                 selection_function=top_n_selection(25),
                 n_genes=None, n_chromosomes=None,
                 elitism=0, p_mutation=0.02,
                 random_state=None,
                 population=None, crossover_points=None):
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

        crossover_points : Indexes used for cross-over points (Default: None, any cross-over point)

        """
        self.fitness_function = fitness_function
        self.selection_function = selection_function
        self.elitism = elitism
        self.p_mutation = p_mutation
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
        # init population
        if population is None:
            self.population_ = self.initialize_population(n_chromosomes, n_genes, self.random_state)
            self.n_genes = n_genes
            self.n_chromosomes = n_chromosomes
        else:
            self.population_, = check_arrays(population)
            self.n_genes = self.population_.shape[1]
            self.n_chromosomes = self.population_.shape[0]
        # crossover points
        if crossover_points is None:
            self.crossover_points = range(self.population_.shape[1])
        else:
            self.crossover_points = crossover_points
        # init fitness
        self.fitness_ = self.fitness()

    def initialize_population(self, n_chromosomes, n_genes, random_state):
        raise Error("initialize_population not implemented")

    def mutate(self, chromosomes, mutation_idx, random_state):
        raise Error("initialize_population not implemented")

    def fitness(self):
        return np.apply_along_axis(self.fitness_function, 1, self.population_)

    def next(self):
        # create new population
        new_population = np.array(self.population_)
        
        # if we have elitism, sort so we have the top the most fit.
        if self.elitism > 0:
            new_population = new_population[np.argsort(self.fitness_)]

        # generate new children
        for i in range(self.elitism, self.n_chromosomes):
            new_population[i] = self.__new_child(self.population_)

        # update pop and fitness
        self.population_ = new_population
        self.fitness_ = self.fitness()

    def best(self, n_best=1):
        f_sorted = np.argsort(self.fitness_)
        p_sorted = self.population_[f_sorted]
        return p_sorted[:n_best], self.fitness_[f_sorted][:n_best]

    def __new_child(self, P_old):
        # choose two random parents
        father_idx, mother_idx = self.selection_function(self.random_state, P_old, self.fitness_)
        father, mother = P_old[father_idx], P_old[mother_idx]
        # breed by single-point crossover
        crossover_idx = self.random_state.choice(self.crossover_points)
        child = np.hstack([father[:crossover_idx], mother[crossover_idx:]])
        # mutate
        mutation_idx = self.random_state.random_sample(child.shape) < self.p_mutation # select which to mutate
        child = self.mutate(child, mutation_idx, self.random_state)
        
        return child

class GeneticAlgorithm(BaseGeneticAlgorithm):
    """
    Continuous genetic algorithm is a genetic algorithm where the gene values
    are chosen from a continous domain. This means the population is randomly
    initialized to [-1.0, 1.0] and mutation modifies the gene value in the range [-1.0, 1.0].
    """
    def __init__(self, scaling=1.0, *args, **kwargs):
        self.scaling = scaling
        super(GeneticAlgorithm, self).__init__(*args, **kwargs)

    def initialize_population(self, n_chromosomes, n_genes, random_state):
        return random_state.rand(n_chromosomes, n_genes) * self.scaling

    def mutate(self, chromosomes, mutation_idx, random_state):
        mutations = (random_state.rand(np.sum(mutation_idx)) - 0.5) * self.scaling
        chromosomes[mutation_idx] += mutations
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
        super(DiscreteGeneticAlgorithm, self).__init__(kwargs)

    def initialize_population(self, n_chromosomes, n_genes, random_state):
        P = np.zeros((n_chromosomes, n_genes))
        for i in range(n_genes):
            P[:,i] = random_state.choice(self.ranges[i], P.shape[0])
        return P

    def mutate(self, chromosomes, mutation_idx, random_state):
        for i in range(self.n_genes):
            midx_i = mutation_idx[:,i]
            chromosomes[:,midx_i] = random_state.choice(self.ranges[i], sum(midx_i))
        return chromosomes
