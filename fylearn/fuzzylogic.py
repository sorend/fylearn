"""
Fuzzy sets and aggregation utils

"""

import numpy as np

# shortcut (but, expensive calc) return max(0.0, min((x - a) / (b - a), (c - x) / (c - b)))
def triangular(a, b, c):
    def f(x):
        if np.isnan(x):
            return 0.0
        elif a < x and x < b:
            return (x - a) / (b - a)
        elif x == b:
            return 1.0
        elif b < x and x < c:
            return (c - x) / (c - b)
        else:
            return 0.0
    # give the specific function
    return f

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
    return pi_f

def prod(x):
    return np.multiply.reduce(x)

def mean(x):
    return np.mean(x)

def lukasiewicz_i(x):
    return max(0, x[0] + x[1] - 1)

def lukasiewicz_u(x):
    return min(1, x[0] + x[1])

def einstein_i(x):
    return (x[0] * x[1]) / (2.0 - (x[0] + x[1] - (x[0] * x[1])))

def einstein_u(x):
    return (x[0] + x[1]) / (1.0 + (x[0] * x[1]))

def algebraic_sum(x):
    return 1.0 - prod(1.0 - np.array(x))

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
