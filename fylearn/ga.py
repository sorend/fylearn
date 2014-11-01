# -*- coding: utf-8 -*-
"""
Genetic algorithm implementation.


"""
import numpy as np
from sklearn.utils import check_arrays

class GeneticAlgorithm:

    def __init__(self, fitness_function, n_genes=None, n_chromosomes=None,
                 keep_parents=10, p_mutation=0.02,
                 scaling=1.0, random_seed=None,
                 population=None, crossover_points=None):
        """
        Initializes the genetic algorithm.

        Parameters:
        -----------

        fitness_function : the function to measure fitness (smaller is more "fit")

        n_genes : number of genes (only needed if population is not specified)

        n_chromosomes : number of chromosomes (only needed if population is not specified)

        keep_parents : number of parents to keep between each generation

        p_mutation : the probability for mutation (applies to each gene)

        scaling : mutation is within [-0.5, 0.5], scaling allows to make the mutation larger/smaller.

        random_seed : the seed for the randomization to use

        population : Initial population, use this or use n_genes and n_chromosomes

        crossover_points : Indexes used for cross-over points (Default: None, any cross-over point)

        """
        self.fitness_function = fitness_function
        self.keep_parents = keep_parents
        self.p_mutation = p_mutation
        self.scaling = scaling
        self.random_seed = random_seed
        # init population
        if population is None:
            self.population_ = np.random.random((n_chromosomes, n_genes)) * self.scaling
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
        self.fitness_ = self._fitness()

    def _fitness(self):
        return np.apply_along_axis(self.fitness_function, 1, self.population_)

    def next(self):
        # sort population by fitness, parents are the top-n
        self.population_ = self.population_[np.argsort(self.fitness_)]
        parents = self.population_[:self.keep_parents]

        # create new children in the remaining elements
        for i in range(self.keep_parents, self.n_chromosomes):
            self.population_[i] = self.__new_child(parents)
        # update fitness
        self.fitness_ = self._fitness()

    def best(self, n_best=1):
        f_sorted = np.argsort(self.fitness_)
        p_sorted = self.population_[f_sorted]
        return p_sorted[:n_best], self.fitness_[f_sorted][:n_best]
        
    def __new_child(self, parents):
        # choose two random parents
        (father, mother) = parents[np.random.choice(len(parents), 2)]
        # breed by single-point crossover
        crossover_idx = np.random.choice(self.crossover_points)
        child = np.hstack([father[:crossover_idx], mother[crossover_idx:]])
        # mutate
        mutation_idx = np.random.random(len(child)) < self.p_mutation # select which to mutate
        mutations = (np.random.random(np.sum(mutation_idx)) - 0.5) * self.scaling # create mutations
        child[mutation_idx] += mutations # add mutations
        
        return child
