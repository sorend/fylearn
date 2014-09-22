# -*- coding: utf-8 -*-


import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_arrays, column_or_1d

from sklearn.utils.graph_shortest_path import graph_shortest_path

import fylearn.fuzzylogic as fl

class KNNClassifier(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {
            "n_neighbors": self.n_neighbors,
            "distance": self.distance
        }

    def set_params(self, **kwargs):
        for key, value in params.items():
            self.setattr(key, value)
        return self

    def __init__(self, n_neighbors=3, distance="euclidean", distance_params={}):
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.distance_params = distance_params

    def fit(self, X, y):
        self._fit_X, = check_arrays(X)
        self.classes_, self._fit_y = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        X, = check_arrays(X)

        # perform along axis 1 (row-wise)
        axis = 1
        
        # calculate distances
        dist = pairwise_distances(X, self._fit_X, self.distance, **self.distance_params)
        
        neigh_ind = np.argpartition(dist, self.n_neighbors-1, axis=axis)
        neigh_ind = neigh_ind[:,:self.n_neighbors]

        # get sorted K
        j = np.arange(neigh_ind.shape[0])[:,None]
        neigh_ind = neigh_ind[j, np.argsort(dist[j, neigh_ind])]

        # get distances from index
        dist = dist[j, neigh_ind]

        # find classes of the neighbors
        classes = self._fit_y[neigh_ind]

        # count max
        u, indices = np.unique(classes, return_inverse=True)
        res = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(neigh_ind.shape),
                                              None, np.max(indices) + 1), axis=axis)]
        
        #print "dist", dist
        #print "neigh_ind", neigh_ind
        #print "classes", classes
        #print "res", res

        return res

class FuzzyKNNClassifier(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {
            "n_neighbors": self.n_neighbors,
            "n_partitions": self.n_partitions,
            "distance": self.distance
        }

    def set_params(self, **kwargs):
        for key, value in params.items():
            self.setattr(key, value)
        return self

    def __init__(self, n_neighbors=3, n_partitions=5, distance="euclidean"):
        self.n_neighbors = n_neighbors
        self.n_partitions = n_partitions
        self.distance = distance

    def fit(self, X, y):
        # update y and save classes.
        self.classes_, self._fit_y = np.unique(y, return_inverse=True)

        self.m, self.n = X.shape
        X, = check_arrays(X)

        # find min/max for each column (axis=0)
        X_min, X_max = np.nanmin(X, axis=0), np.nanmax(X, axis=0)

        # calculate membership of an element
        def to_fuzzy(mu_list):
            def to_fuzzy_curried(x):
                return np.argmax([ mu(x) for mu in mu_list ])
            return to_fuzzy_curried

        def drange(start, stop, step):
            r = start
            while r < stop:
                yield r
                r += step

        # build membership functions
        self.mus_ = {}
        # initialize fuzzy partition
        self._fit_X = np.zeros((self.m, self.n * self.n_partitions))
        #
        for idx, (x_min, x_max) in enumerate(zip(X_min, X_max)):
            x_diff = x_max - x_min
            x_part = x_diff / self.n_partitions

            # create mus and vectorized converter
            mus = [ fl.triangular(x-x_part, x, x+x_part) for x in drange(x_min, x_max, x_part) ]
            self.mus_[idx] = mus

            for mu_idx, mu in enumerate(mus):
                self._fit_X[:,(idx*self.n_partitions)+mu_idx] = np.vectorize(mu)(X[:,idx])

        print "shape(o)", self.m, "x", self.n
        print "shape(f)", self._fit_X.shape
        
        return self

    def pairwise_distances(self, X_f, Y_f):

        D = np.zeros((len(X_f), len(Y_f)))

        # apply prod
        for row_idx, x_i in enumerate(X_f):
            Z_f = x_i * Y_f

            F = np.zeros((self.m, self.n))
            for idx in range(self.n):
                offset = idx * self.n_partitions
                tmp = Z_f[:,offset:offset+self.n_partitions]
                F[:,idx] = np.amax(Z_f[:,offset:offset+self.n_partitions], axis=1)
            D[row_idx,:] = np.mean(F, axis=1)

        return D
    
    def predict(self, X):
        X, = check_arrays(X)

        # convert to fuzzy
        X_f = np.zeros((X.shape[0], self.n_partitions * X.shape[1]))
        for idx, mus in self.mus_.items():
            for mu_idx, mu in enumerate(mus):
                X_f[:,(idx*self.n_partitions)+mu_idx] = np.vectorize(mu)(X[:,idx])

        # perform along axis 1 (row-wise)
        axis = 1 
       
        # calculate distances
        dist = self.pairwise_distances(X_f, self._fit_X)
        
        neigh_ind = np.argpartition(dist, self.n_neighbors-1, axis=axis)
        neigh_ind = neigh_ind[:,:self.n_neighbors]

        # get sorted K
        j = np.arange(neigh_ind.shape[0])[:,None]
        neigh_ind = neigh_ind[j, np.argsort(dist[j, neigh_ind])]

        # get distances from index
        dist = dist[j, neigh_ind]

        # find classes of the neighbors
        classes = self._fit_y[neigh_ind]

        # count max
        u, indices = np.unique(classes, return_inverse=True)
        res = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(neigh_ind.shape),
                                              None, np.max(indices) + 1), axis=axis)]
        
        return res


class CentralityKNNClassifier(BaseEstimator, ClassifierMixin):

    def get_params(self, deep=False):
        return {
            "n_neighbors": self.n_neighbors,
            "distance": self.distance
        }

    def set_params(self, **kwargs):
        for key, value in params.items():
            self.setattr(key, value)
        return self

    def __init__(self, n_neighbors=3, distance="euclidean", distance_params={}):
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.distance_params = distance_params

    def fit(self, X, y):
        self._fit_X, = check_arrays(X)
        self.classes_, self._fit_y = np.unique(y, return_inverse=True)

        # calculate distances
        dist = pairwise_distances(self._fit_X, self._fit_X, self.distance, **self.distance_params)

        sp = graph_shortest_path(dist)

        print dist
        print dist.shape
        print sp
        print "min", np.min(sp), "max", np.max(sp), "mean", np.mean(sp)
        
        return self

    def predict(self, X):
        X, = check_arrays(X)

        # perform along axis 1 (row-wise)
        axis = 1
        
        # calculate distances
        dist = pairwise_distances(X, self._fit_X, self.distance, **self.distance_params)
        
        neigh_ind = np.argpartition(dist, self.n_neighbors-1, axis=axis)
        neigh_ind = neigh_ind[:,:self.n_neighbors]

        # get sorted K
        j = np.arange(neigh_ind.shape[0])[:,None]
        neigh_ind = neigh_ind[j, np.argsort(dist[j, neigh_ind])]

        # get distances from index
        dist = dist[j, neigh_ind]

        # find classes of the neighbors
        classes = self._fit_y[neigh_ind]

        # count max
        u, indices = np.unique(classes, return_inverse=True)
        res = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(neigh_ind.shape),
                                              None, np.max(indices) + 1), axis=axis)]
        
        #print "dist", dist
        #print "neigh_ind", neigh_ind
        #print "classes", classes
        #print "res", res

        return res
