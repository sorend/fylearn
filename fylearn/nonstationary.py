# -*- coding: utf-8 -*-
"""
Implementation of non-stationary fuzzy sets [1]

A nonstationary fuzzy set can be described as a fuzzy set which changes
its parameters with a timestamp t.

$A = \int_{t \in T} \int_{x \in X} \mu_A(t, x)/x/t$

For a function having $m$ parameters, $p \in \{1, \dots, m\}$, we can think of its definition as

$\mu_A(t, x) = \mu_A(x, p_1(t), \dots, p_m(t))$

We allow the parameters to vary around the actual membership function, based on the timestamp $t$.
The basic idea is to allow minor variations in the membership function, not to allow it to change
drastically, which is up to the pertubation function.

For each call to the membership function, $t$ will be increased (think of it as seconds).

[1] Garibaldi, 2007.

"""

import numpy as np

def helper_stationary_value(val):
    """
    Helper for creating a parameter which is stationary (not changing based on t)
    in a stationary fuzzy set. Returns the static pertubation function

    Parameters:
    -----------

    val : the stationary value
    """
    return lambda t: val

class NonstationaryFuzzySet:

    def __init__(self, factory, **pertubations):
        """
        Initializes the nonstationary fuzzy set.

        Parameters:
        -----------

        factory : the factory method to create the backing set of the nonstationary set.

        pertubations : pertubation functions for the arguments of the backing set. Note each
                       parameter for the backing set must be defined by a pertubation function.
        """
        self.factory = factory
        self.pertubations = pertubations

    def __call__(self, T, X):
        """
        Calculates membership values of a row of values of X for a given set of T.

        Parameters:
        -----------

        T : The set of timestamps in the nonstationary set to calculate for

        X : The values to calculate membership functions for, each row in X
            corresponds to one time value,
        """
        # ensure 2d
        X = np.array(X)
        if X.ndim == 1:
            X = np.array([ X ])

        T = np.atleast_1d(T)

        if T.ndim != 1:
            raise ValueError("T must be 1-dimensional array")

        if len(T) != len(X):
            raise ValueError("len(T) != len(X): Rows in X must match timestamps in T")

        Y = np.zeros(X.shape)

        # calculate for each timestamp
        for idx, t in enumerate(T):
            # build set
            params = { k: v(t) for k, v in self.pertubations.items() }
            mu = self.factory(**params)
            # calculate values
            Y[idx, ] = mu(X[idx, ])

        return Y
