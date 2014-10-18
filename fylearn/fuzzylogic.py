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

    def __str__(self):
        return "Δ(%.2f %.2f %.2f)" % (self.a, self.b, self.c)

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
        return "pi(%.2f %.2f %.2f)" % (self.a, self.r, self.b)
    
# pi shaped function (bell)
def pi(a, r, b, m=2.0):
    p = (r + a) / 2.0
    q = (b + r) / 2.0
    def pi_f(x):
        if x <= a:
            return 0.0
        elif a < x and x <= p:
            return (2 ** (m - 1)) * (((x - a) / (r - a)) ** m)
        elif p < x and x <= r:
            return 1.0 - ((2 ** (m - 1)) * (((r - x) / (r - a)) ** m))
        elif r < x and x <= q:
            return 1.0 - ((2 ** (m - 1)) * (((x - r) / (b - r)) ** m))
        elif q < x and x <= b:
            return ((2 ** (m - 1)) * (((b - x) / (b - r)) ** m))
        else:
            return 0.0
    # save factory values
    pi_f.a = a
    pi_f.r = r
    pi_f.b = b
    pi_f.m = m
    pi_f.__str__ = lambda: "pi(%.2f %.2f %.2f)" % (pi_f.a, pi_f.r, pi_f.b)
    return pi_f

def prod(X):
    X = np.array(X)
    d = len(X.shape)
    return np.multiply.reduce(X, d-1)

def mean(X):
    X = np.array(X)
    d = len(X.shape)
    return np.mean(X, d-1)

def lukasiewicz_i(x):
    return max(0, x[0] + x[1] - 1)

def lukasiewicz_u(x):
    return min(1, x[0] + x[1])

def einstein_i(x):
    return (x[0] * x[1]) / (2.0 - (x[0] + x[1] - (x[0] * x[1])))

def einstein_u(x):
    return (x[0] + x[1]) / (1.0 + (x[0] * x[1]))

def algebraic_sum(X):
    return 1.0 - prod(1.0 - np.array(X))

def owa(w):
    w_a = np.array(w)
    def owa_f(x):
        c = np.max(len(x), len(w_a))
        w_a = w_a[:c]
        s = np.sort(x)[:c]
        return np.sum(s * w_a)
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
