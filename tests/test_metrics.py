

from fylearn.metrics import CLUSTER_DISTANCE_METHODS, dunn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

import pytest


def test_diameter_methods_mean_cluster():

    data = load_iris()
    kmeans = KMeans(n_clusters=3)
    c = data['target']
    x = data['data']
    k = kmeans.fit_predict(x)
    d = euclidean_distances(x)

    cdist_method = CLUSTER_DISTANCE_METHODS[0]

    dund = dunn(c, d, 'mean_cluster', cdist_method)
    dunk = dunn(k, d, 'mean_cluster', cdist_method)

    print('mean_cluster', cdist_method, dund, dunk)

    assert pytest.approx(0.0084, abs=0.0001) == dunk
    assert pytest.approx(0.0077, abs=0.0001) == dund


def test_diameter_methods_farthest():

    data = load_iris()
    kmeans = KMeans(n_clusters=3)
    c = data['target']
    x = data['data']
    k = kmeans.fit_predict(x)
    d = euclidean_distances(x)

    cdist_method = CLUSTER_DISTANCE_METHODS[0]

    dund = dunn(c, d, 'farthest', cdist_method)
    dunk = dunn(k, d, 'farthest', cdist_method)

    print('farthest', cdist_method, dund, dunk)

    assert pytest.approx(0.098, abs=0.001) == dunk
    assert pytest.approx(0.058, abs=0.001) == dund
