# -*- coding: utf-8 -*-
"""
Fuzzy sets and aggregation utils

"""

#
# Author: Soren A. Davidsen <sorend@gmail.com>
#

import numpy as np
import collections
import numbers
from scipy.optimize import minimize

def helper_np_array(X):
    if isinstance(X, (np.ndarray, np.generic)):
        return X
    elif isinstance(X, collections.Sequence):
        return np.array(X)
    elif isinstance(X, numbers.Number):
        return np.array([X])
    else:
        raise ValueError("unsupported type for building np.array: %s" % (type(X),))

class TriangularSet:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, X):
        X = helper_np_array(X)
        y = np.zeros(X.shape)  # allocate output (y)
        left = (self.a < X) & (X < self.b)  # find where to apply left
        right = (self.b < X) & (X < self.c)  # find where to apply right
        y[left] = (X[left] - self.a) / (self.b - self.a)
        y[X == self.b] = 1.0  # at top
        y[right] = (self.c - X[right]) / (self.c - self.b)
        return y

    def __str__(self):
        return "Δ(%.2f %.2f %.2f)" % (self.a, self.b, self.c)

    def __repr__(self):
        return str(self)

class TrapezoidalSet:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, X):
        X = helper_np_array(X)
        y = np.zeros(X.shape)
        left = (self.a < X) & (X < self.b)
        center = (self.b <= X) & (X <= self.c)
        right = (self.c < X) & (X < self.d)
        y[left] = (X[left] - self.a) / (self.b - self.a)
        y[center] = 1.0
        y[right] = (self.d - X[right]) / (self.d - self.c)
        return y

    def __str__(self):
        return "T(%.2f %.2f %.2f %.2f)" % (self.a, self.b, self.c, self.d)

class PiSet:
    def __init__(self, r, a=None, b=None, p=None, q=None, m=2.0):

        if a is not None:
            self.a = a
            self.p = (r + a) / 2.0  # between r and a
        elif p is not None:
            self.p = p
            self.a = r - (2.0 * (r - p))  # one "p" extra.
        else:
            raise ValueError("please specify a or p")

        if b is not None:
            self.b = b
            self.q = (r + b) / 2.0
        elif q is not None:
            self.q = q
            self.b = r + (2.0 * (q - r))
        else:
            raise ValueError("please specify b or q")

        # if a >= r or r >= b:
        #     raise ValueError("please ensure a < r < b, got: a=%f, r=%f b=%f" % (self.a, self.r, self.b))

        self.r = r
        self.m = m
        self.S = (2 ** (m - 1.0))

        self.r_a = self.r - self.a
        self.b_r = self.b - self.r

    def __call__(self, X):
        X = helper_np_array(X)

        y = np.zeros(X.shape)

        l1 = (self.a < X) & (X <= self.p)  # left lower
        l2 = (self.p < X) & (X <= self.r)  # left upper
        r1 = (self.r < X) & (X <= self.q)  # right upper
        r2 = (self.q < X) & (X <= self.b)  # right lower

        y[l1] = self.S * (((X[l1] - self.a) / (self.r_a)) ** self.m)
        y[l2] = 1.0 - (self.S * (((self.r - X[l2]) / (self.r_a)) ** self.m))
        y[r1] = 1.0 - (self.S * (((X[r1] - self.r) / (self.b_r)) ** self.m))
        y[r2] = self.S * (((self.b - X[r2]) / (self.b_r)) ** self.m)

        return y

    def __str__(self):
        return "π(%.2f %.2f %.2f)" % (self.p, self.r, self.q)

    def __repr__(self):
        return str(self)

def prod(X, axis=-1):
    """Product along dimension 0 or 1 depending on array or matrix"""
    return np.multiply.reduce(X, axis)

def mean(X, axis=-1):
    return np.nanmean(X, axis)

def min(X, axis=-1):
    return np.nanmin(X, axis)

def max(X, axis=-1):
    return np.nanmax(X, axis)

def lukasiewicz_i(X):
    return np.maximum(0.0, X[:, 0] + X[:, 1] - 1)

def lukasiewicz_u(X):
    return np.minimum(1.0, X[:, 0] + X[:, 1])

def einstein_i(X):
    a, b = X[:, 0], X[:, 1]
    return (a * b) / (2.0 - (a + b - (a * b)))

def einstein_u(X):
    a, b = X[:, 0], X[:, 1]
    return (a + b) / (1.0 + (a * b))

def algebraic_sum(X, axis=-1):
    return 1.0 - prod(1.0 - X, axis)

def min_max_normalize(X):
    nmin, nmax = np.nanmin(X), np.nanmax(X)
    return (X - nmin) / (nmax - nmin)

def p_normalize(X, axis=None):

    s = np.sum(X, axis=axis, dtype="float")

    def fixzero(x):
        return 1.0 if x == 0.0 else x

    def afixzero(x):
        x[x == 0.0] = 1.0
        return x

    if axis is None:
        return X / fixzero(s)
    elif axis == 0:
        return X / afixzero(s)
    elif axis == 1:
        return (X.T / afixzero(s)).T
    else:
        raise ValueError("axis must be None or 0 or 1")

def dispersion(w):
    return -np.sum(w[w > 0.0] * np.log(w[w > 0.0]))  # filter 0 as 0 * -inf is undef in NumPy

def ndispersion(w):
    return dispersion(w) / np.log(len(w))

def yager_orness(w):
    """
    The orness is a measure of how "or-like" a given weight vector is for use in OWA.

    orness(w) = 1/(n-1) * sum( (n-i)*w )
    """
    n = len(w)
    return np.sum(np.arange(n - 1, -1, -1) * w) / (n - 1.0)

def yager_andness(w):
    """
    Yager's andness is 1.0 - Yager's orness for a given weight vector.
    """
    return 1.0 - yager_orness(w)

def weights_mapping(w):
    s = np.e ** w
    return s / np.sum(s)

class OWA(object):
    """
    Order weighted averaging operator.

    The order weighted averaging operator aggregates vector of a1, ..., an using a
    a permutation b1, ... bn for which b1 >= b2 => ... >= bn and a weight vector
    w, for which that w = w1, ..., wn in [0, 1] and sum w = 1

    Averaging is done with weighted mean: sum(b*w)

    Parameters:
    -----------
    v : The weights

    """
    def __init__(self, v):
        self.v = v
        self.v_ = v[::-1]  # save the inverse so we don't need to reverse np.sort
        self.lv = len(v)

    def __call__(self, X, axis=-1):
        if X.shape[axis] != self.lv:
            raise ValueError("len(X) != len(v)")
        b = np.sort(X, axis)  # creates permutation
        return self.sorted_mean(b, axis)

    def sorted_mean(self, X, axis=-1):
        """Use for pre-sorted X"""
        return np.sum(X * self.v_, axis)

    def __str__(self):
        return "OWA(" + " ".join([ "%.4f" % (x,) for x in self.v]) + ")"

    def __repr__(self):
        return str(self)

    def andness(self):
        return yager_andness(self.v)

    def orness(self):
        return yager_orness(self.v)

    def disp(self):
        return dispersion(self.v)

    def ndisp(self):
        return ndispersion(self.v)

class GOWA(OWA):
    """
    Generalized order weighted averaging operator.

    The generalized order weighted averaging operator aggregates a vector of a1, ..., an using
    a permutation b1, ..., bn for which b1 >= b2 => ... => bn where b1 is the largest value in
    a, and which has an related weight vector w = w1, ..., wn in [0, 1] and sum w = 1.

    Averaging is used using the power-mean: sum(w*b^p)^(1/p), where p is the power parameter.
    """
    def __init__(self, p, v):
        """
        Constructs GOWA operator.

        Parameters:
        -----------
        p : power parameter.

        v : weights.
        """
        super(GOWA, self).__init__(v)
        self.p = p
        self.inv_p = (1.0 / p)

    def sorted_mean(self, X, axis=-1):
        return np.sum((X ** self.p) * self.v_, axis) ** self.inv_p

    def __str__(self):
        return ("GOWA(%f, " % (self.p,)) + " ".join([ "%.4f" % (x,) for x in self.v]) + ")"

def gowa(*w):
    """Create Generalized OWA (GOWA) operator from weights"""
    w = np.array(w, copy=False).ravel()
    return GOWA(w)

def owa(*w):
    """Create OWA operator from weights"""
    w = np.array(w, copy=False).ravel()
    return OWA(w)

def meowa(n, orness, **kwargs):
    """
    Maximize dispersion at a specified orness level.
    """
    if 0.0 > orness or orness > 1.0:
        raise ValueError("orness must be in [0, 1]")

    if n < 2:
        raise ValueError("n must be > 1")

    def negdisp(v):
        return -dispersion(v)  # we want to maximize, but scipy want to minimize

    def constraint_has_orness(v):
        return yager_orness(v) - orness

    def constraint_has_sum(v):
        return np.sum(v) - 1.0

    return _minimize_owa(negdisp, (constraint_has_orness, constraint_has_sum), n, **kwargs)

def sampling_owa_orness(x, d, **kwargs):
    """
    Maximize orness of an owa operator given a given result data point.
    """
    n = len(x)
    if n < 2:
        raise ValueError("n must be > 1")

    s_ = np.sort(x)[::-1]

    def negorness(v):
        return -yager_orness(v)

    def constraint_has_output_d(v):
        return np.sum(s_ * v) - d

    def constraint_has_sum(v):
        return np.sum(v) - 1.0

    return _minimize_owa(negorness, (constraint_has_sum, constraint_has_output_d), n, **kwargs)

def sampling_owa_ndisp(x, d, **kwargs):
    """
    Maximize dispersion of an owa operator given a given result data point.
    """
    n = len(x)
    if n < 2:
        raise ValueError("n must be > 1")

    s_ = np.sort(x)[::-1]

    def negndisp(v):
        return -ndispersion(v)

    def constraint_has_output_d(v):
        return np.sum(s_ * v) - d

    def constraint_has_sum(v):
        return np.sum(v) - 1.0

    return _minimize_owa(negndisp, (constraint_has_sum, constraint_has_output_d), n, **kwargs)

def mvowa(n, orness, **kwargs):
    """
    Maximum variability order weighted aggregation. Construct aggregation with fixed orness
    but maximized variance. [Fuller and Majlender, 2003]
    """
    if 0.0 > orness or orness > 1.0:
        raise ValueError("orness must be in [0, 1]")

    if n < 2:
        raise ValueError("n must be > 1")

    def variance(v):
        n = len(v)
        return (np.sum(v * v) - (1 - (n * n))) / n

    def constraint_has_orness(v):
        return yager_orness(v) - orness

    def constraint_has_sum(v):
        return np.sum(v) - 1.0

    return _minimize_owa(variance, (constraint_has_orness, constraint_has_sum), n, **kwargs)

def _minimize_owa(minfunc, constraints, n, **kwargs):

    bounds = tuple([ (0, 1) for x in range(n) ])  # this is actually the third constraint, but common.

    initial = np.ones(n) / n

    constraints_ = tuple([ {"fun": c, "type": "eq"} for c in constraints ])

    res = minimize(minfunc, initial,
                   bounds=bounds,
                   options=kwargs,
                   constraints=constraints_)

    if res.success:
        return OWA(res.x)
    else:
        raise ValueError("Could not optimize weights: " + res.message)

class AndnessDirectedAveraging:
    def __init__(self, p):
        self.p = p
        self.tnorm = p <= 0.5
        self.alpha = (1.0 - p) / p if self.tnorm else p / (1.0 - p)

    def __call__(self, X, axis=-1):
        X = np.array(X, copy=False)
        if self.tnorm:
            return (np.sum(X ** self.alpha, axis) / X.shape[axis]) ** (1.0 / self.alpha)
        else:
            return 1.0 - ((np.sum((1.0 - X) ** (1.0 / self.alpha), axis) / X.shape[axis]) ** self.alpha)

def aa(p):
    assert 0 < p and p < 1
    return AndnessDirectedAveraging(p)
