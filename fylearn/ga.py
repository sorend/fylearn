# -*- coding: utf-8 -*-
"""
Genetic algorithm implementation

"""
import numpy as np

class GeneticAlgorithm:

    def __init__(self, n_genes, n_chromosomes,
                 fitness_function,
                 keep_parents=10, p_mutation=0.02,
                 scaling=10.0, random_seed=None):
        self.n_genes = n_genes
        self.n_chromosomes = n_chromosomes
        self.fitness_function = fitness_function
        self.keep_parents = keep_parents
        self.p_mutation = p_mutation
        self.scaling = scaling
        self.random_seed = random_seed
        # init population
        self.population_ = (np.random.random((self.n_chromosomes, self.n_genes)) - 0.5) * self.scaling
        # init fitness
        self.fitness_ = np.apply_along_axis(self.fitness_function, 1, self.population_)

    def next(self):
        # sort population based on 
        f_sorted = np.argsort(self.fitness_)
        self.population_ = self.population_[f_sorted]
        # parents
        parents = self.population_[:self.keep_parents]
        
        # create new children
        for i in range(self.keep_parents, self.n_chromosomes):
            self.population_[i] = self.new_child(parents)
        # update fitness
        self.fitness_ = np.apply_along_axis(self.fitness_function, 1,
                                            self.population_)

    def new_child(self, parents):
        # choose parents
        (father, mother) = parents[np.random.choice(len(parents), 2)]
        # breed by single-point crossover
        crossover_idx = np.random.choice(len(father))
        child = np.hstack([father[:crossover_idx], mother[crossover_idx:]])
        # mutate
        mutation_idx = np.random.random(len(child)) < self.p_mutation # select which to mutate
        mutations = (np.random.random(np.sum(mutation_idx)) - 0.5) * self.scaling # create mutations
        child[mutation_idx] += mutations # add mutations

        return child

if __name__ == "__main__":
    # fitness function is the variance (means, prefer with small variance)
    ff = lambda x: np.var(x)
    # create instance
    ga = GeneticAlgorithm(5, 100, ff, keep_parents=10, p_mutation=0.1)

    for i in range(100):
        ga.next()
        print "fitness", ga.fitness_[:4]

    print ga.population_[0]
