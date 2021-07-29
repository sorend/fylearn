# -*- coding: utf-8 -*-
"""Additional metrics used by the library.

The module contains the following metrics:
- Dunn index

### References:

[1] Kovács, F., Legány, C., & Babos, A. (2005). Cluster validity measurement techniques. 6th International Symposium of
    Hungarian Researchers on Computational Intelligence.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

DIAMETER_METHODS = ['mean_cluster', 'farthest']
CLUSTER_DISTANCE_METHODS = ['nearest', 'farthest']


def inter_cluster_distances(labels, distances, method='nearest'):
    """Calculate distances between the two nearest points of each cluster.

    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: `nearest` for the distances between the two nearest points in each cluster, or `farthest`
    """
    if method not in CLUSTER_DISTANCE_METHODS:
        raise ValueError(
            'method must be one of {}'.format(CLUSTER_DISTANCE_METHODS))

    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                (not farthest and
                 distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                    or
                (farthest and
                 distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
    """Calculate cluster diameters

    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: either `mean_cluster` for the mean distance between all elements in each cluster, or `farthest` for the distance between the two points furthest from each other
    """
    if method not in DIAMETER_METHODS:
        raise ValueError('method must be one of {}'.format(DIAMETER_METHODS))

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[
                        labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn(labels, distances, diameter_method='farthest', cdist_method='nearest'):
    r"""Dunn index for cluster validation (larger is better).

    .. math:: D = \min_{i = 1 \ldots n_c; j = i + 1\ldots n_c} \left\lbrace
       \frac{d \left( c_i,c_j \right)}{\max_{k = 1 \ldots n_c} \left(diam \left(c_k \right) \right)} \right\rbrace

    where :math:`d(c_i,c_j)` represents the distance between
    clusters :math:`c_i` and :math:`c_j`, and :math:`diam(c_k)` is the diameter of cluster :math:`c_k`.
    Inter-cluster distance can be defined in many ways, such as the distance between cluster centroids or between
    their closest elements. Cluster diameter can be defined as the mean distance between all elements in the cluster,
    between all elements to the cluster centroid, or as the distance between the two furthest elements.
    The higher the value of the resulting Dunn index, the better the clustering
    result is considered, since higher values indicate that clusters are
    compact (small :math:`diam(c_k)`) and far apart (large :math:`d \left( c_i,c_j \right)`).

    Parameters:
    -----------
    labels : a list containing cluster labels for each of the n elements
    distances : an n x n numpy.array containing the pairwise distances between elements
    diameter_method : diameter method, see :py:function:`diameter` `method` parameter
    cdist_method : cluster distance method, see :py:function:`diameter` `method` parameter
    """
    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameter(labels, distances, diameter_method))

    return min_distance / max_diameter


if __name__ == '__main__':
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans

    data = load_iris()
    kmeans = KMeans(n_clusters=3)
    c = data['target']
    x = data['data']
    k = kmeans.fit_predict(x)
    d = euclidean_distances(x)

    for diameter_method in DIAMETER_METHODS:
        for cdist_method in CLUSTER_DISTANCE_METHODS:
            dund = dunn(c, d, diameter_method, cdist_method)
            dunk = dunn(k, d, diameter_method, cdist_method)
            print(diameter_method, cdist_method, dund, dunk)
