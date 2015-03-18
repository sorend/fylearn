# -*- coding: utf-8 -*-
"""
Fuzzy sets and aggregation utils

"""

import numpy as np

class TriangularSet:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, X):
        y = np.zeros(X.shape) # allocate output (y)
        left  = (self.a < X) & (X < self.b) # find where to apply left
        right = (self.b < X) & (X < self.c) # find where to apply right
        y[left] = (X[left] - self.a) / (self.b - self.a)
        y[X == self.b] = 1.0 # at top
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
        y = np.zeros(X.shape)
        left   = (self.a < X) & (X < self.b)
        center = (self.b <= X) & (X <= self.c)
        right  = (self.c < X) & (X < self.d)
        y[left] = (X[left] - self.a) / (self.b - self.a)
        y[center] = 1.0
        y[right] = (self.d - X[right]) / (self.d - self.c)
        return y

    def __str__(self):
        return "T(%.2f %.2f %.2f %.2f)" % (self.a, self.b, self.c, self.d)

class PiSet:
    def __init__(self, a, r, b, m=2.0):
        self.a = a
        self.r = r
        self.b = b
        self.m = m
        self.p = (r + a) / 2.0
        self.q = (b + r) / 2.0
        self.S = (2**(m - 1.0))

    def __call__(self, X):
        y = np.zeros(X.shape)

        l1 = (self.a < X) & (X <= self.p) # left lower
        l2 = (self.p < X) & (X <= self.r) # left upper
        r1 = (self.r < X) & (X <= self.q) # right upper
        r2 = (self.q < X) & (X <= self.b) # right lower

        y[l1] = self.S * (((X[l1] - self.a) / (self.r - self.a)) ** self.m)
        y[l2] = 1.0 - (self.S * (((self.r - X[l2]) / (self.r - self.a)) ** self.m))
        y[r1] = 1.0 - (self.S * (((X[r1] - self.r) / (self.b - self.r)) ** self.m))
        y[r2] = self.S * (((self.b - X[r2]) / (self.b - self.r)) ** self.m)

        return y

    def __str__(self):
        return "π(%.2f %.2f %.2f)" % (self.p, self.r, self.q)

    def __repr__(self):
        return str(self)

def prod(X):
    """Product along dimension 0 or 1 depending on array or matrix"""
    return np.multiply.reduce(X, -1)

def mean(X):
    return np.mean(X, -1)

def min(X):
    return np.nanmin(X, -1)

def max(X):
    return np.nanmin(X, -1)

def lukasiewicz_i(X):
    return np.maximum(0.0, X[:,0] + X[:,1] - 1)

def lukasiewicz_u(X):
    return np.minimum(1.0, X[:,0] + X[:,1])

def einstein_i(X):
    a, b = X[:,0], X[:,1]
    return (a * b) / (2.0 - (a + b - (a * b)))

def einstein_u(X):
    a, b = X[:,0], X[:,1]
    return (a + b) / (1.0 + (a * b))

def algebraic_sum(X):
    return 1.0 - prod(1.0 - X)

def owa(*w):
    w = np.array(w, copy=False).ravel()
    def owa_f(X):
        v = w[::-1]
        lv = len(v)
        if X.shape[-1] < lv:
            raise Exception("len(X) < len(w)")
        elif X.shape[-1] > lv:
            missing = [ 0.0 ] * (X.shape[-1] - lv)
            v = np.append(missing, v)
        return np.sum(np.sort(X, -1) * v, -1)
    return owa_f

def aiwa(p):
    assert 0 < p and p < 1
    alpha = (1.0 - p) / p
    if p <= 0.5:
        def aiwa_f_t(x):
            x = np.array(x)
            return (np.sum(x ** 2) / len(x)) ** (1 / alpha)
        return aiwa_f_t
    else:
        def aiwa_f_co(x):
            x = np.array(x)
            return 1.0 - ((np.sum((1.0 - x) ** (1 / alpha)) / len(x)) ** alpha)
        return aiwa_f_co
