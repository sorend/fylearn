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
    return np.mean(X, axis)

def min(X, axis=-1):
    return np.nanmin(X, axis)

def max(X, axis=-1):
    return np.nanmin(X, axis)

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

def algebraic_sum(X):
    return 1.0 - prod(1.0 - X)

def min_max_normalize(X):
    nmin, nmax = np.nanmin(X), np.nanmax(X)
    return (X - nmin) / (nmax - nmin)

def p_normalize(X):
    return X / np.sum(X)

def dispersion(w):
    return -np.sum(w[w > 0.0] * np.log(w[w > 0.0]))  # filter 0 as 0 * -inf is undef in NumPy

class OWA:
    def __init__(self, v):
        self.v = v
        self.lv = len(v)

    def __call__(self, X, axis=-1):
        v, lv = self.v, self.lv
        if X.shape[axis] < lv:
            raise ValueError("len(X) < len(w)")
        elif X.shape[axis] > lv:
            missing = [ 0. ] * (X.shape[axis] - lv)
            v = np.append(missing, v)
        return np.sum(np.sort(X, axis) * v, axis)

    def __str__(self):
        return "OWA(v=%s)" % (str(self.v),)

    def andness(self):
        return 1.0 - self.orness()

    def orness(self):
        v = self.v
        n = self.lv
        return np.sum(np.array([ n - (i + 1) for i in range(n) ]) * v) / (n - 1.0)

    def disp(self):
        return dispersion(self.v)

    def ndisp(self):
        return dispersion(self.v) / np.log(self.lv)

def owa(*w):
    w = np.array(w, copy=False).ravel()
    return OWA(w[::-1])

def meowa(n, rho):
    if 0.0 > rho or rho > 1.0:
        raise ValueError("rho must be in [0, 1]")

    if rho == 1.0:
        w = np.zeros(n)
        w[-1] = 1.0
        return OWA(w)

    # calc roots
    t_m = (-(rho - 0.5) - np.sqrt(((rho - 0.5) * (rho - 0.5)) - (4 * (rho - 1.0) * rho))) / (2.0 * (rho - 1.0))
    t_p = (-(rho - 0.5) + np.sqrt(((rho - 0.5) * (rho - 0.5)) - (4 * (rho - 1.0) * rho))) / (2.0 * (rho - 1.0))
    t = np.maximum(t_m, t_p)

    w = np.array([ np.power(t, i) for i in range(n) ])

    w = p_normalize(w)

    return OWA(w)

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
