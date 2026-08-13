"""
Fuzzy sets and aggregation utils

"""

#
# Author: Soren A. Davidsen <sorend@gmail.com>
#

import numbers
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def helper_np_array(X: Any) -> Any:
    if isinstance(X, (np.ndarray, np.generic)):
        return X
    elif isinstance(X, Sequence):
        return np.array(X)
    elif isinstance(X, numbers.Number):
        return np.array([X])
    else:
        raise ValueError(f"unsupported type for building np.array: {type(X)}")


class ZadehNegatedSet:
    def __init__(self, s: Callable[[Any], Any]):
        self.s = s

    def __call__(self, X: Any) -> Any:
        return 1.0 - self.s(X)

    def __str__(self) -> str:
        return f"Not({str(self.s)})"


class TriangularSet:
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, X: Any) -> np.ndarray:
        X = helper_np_array(X)
        y = np.zeros(X.shape)  # allocate output (y)
        left = (self.a < X) & (X < self.b)  # find where to apply left
        right = (self.b < X) & (X < self.c)  # find where to apply right
        y[left] = (X[left] - self.a) / (self.b - self.a)
        y[X == self.b] = 1.0  # at top
        y[right] = (self.c - X[right]) / (self.c - self.b)
        return y

    def __str__(self) -> str:
        return f"Δ({self.a:.2f} {self.b:.2f} {self.c:.2f})"

    def __repr__(self) -> str:
        return str(self)


class TrapezoidalSet:
    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, X: Any) -> np.ndarray:
        X = helper_np_array(X)
        y = np.zeros(X.shape)
        left = (self.a < X) & (X < self.b)
        center = (self.b <= X) & (X <= self.c)
        right = (self.c < X) & (X < self.d)
        y[left] = (X[left] - self.a) / (self.b - self.a)
        y[center] = 1.0
        y[right] = (self.d - X[right]) / (self.d - self.c)
        return y

    def __str__(self) -> str:
        return f"T({self.a:.2f} {self.b:.2f} {self.c:.2f} {self.d:.2f})"


class PiSet:
    def __init__(
        self,
        r: float,
        a: float | None = None,
        b: float | None = None,
        p: float | None = None,
        q: float | None = None,
        m: float = 2.0,
    ):
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
        self.S = 2 ** (m - 1.0)

        self.r_a = self.r - self.a
        self.b_r = self.b - self.r

    def __call__(self, X: Any) -> np.ndarray:
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

    def __str__(self) -> str:
        return f"π(p={self.p:.2f} r={self.r:.2f} q={self.q:.2f})"

    def __repr__(self) -> str:
        return str(self)


def prod(X: np.ndarray, axis: int = -1) -> np.ndarray:
    """Product along dimension 0 or 1 depending on array or matrix"""
    return np.multiply.reduce(X, axis)


def mean(X: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.nanmean(X, axis)


def min(X: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.nanmin(X, axis)


def max(X: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.nanmax(X, axis)


def lukasiewicz_i(X: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, X[:, 0] + X[:, 1] - 1)


def lukasiewicz_u(X: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, X[:, 0] + X[:, 1])


def einstein_i(X: np.ndarray) -> np.ndarray:
    a, b = X[:, 0], X[:, 1]
    return (a * b) / (2.0 - (a + b - (a * b)))


def einstein_u(X: np.ndarray) -> np.ndarray:
    a, b = X[:, 0], X[:, 1]
    return (a + b) / (1.0 + (a * b))


def algebraic_sum(X: np.ndarray, axis: int = -1) -> np.ndarray:
    return 1.0 - prod(1.0 - X, axis)


def min_max_normalize(X: np.ndarray) -> np.ndarray:
    nmin, nmax = np.nanmin(X), np.nanmax(X)
    return (X - nmin) / (nmax - nmin)


def p_normalize(X: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Normalize values as probabilities (sums to one)

    Parameters:
    -----------

    X : the numpy array to normalize

    axis : None does not deal with axes (default), 0: probas by row-sums, 1: probas by column-sums
    """
    assert axis in [None, 0, 1], "Only axes None, 0 and 1 is supported"

    def handle_all_zeros(a: np.ndarray) -> np.ndarray:
        b = np.sum(X, dtype="float")
        if b > 0.0:
            return a / b
        else:
            return np.ones(a.shape) / a.size

    def handle_zero_rows(a: np.ndarray) -> np.ndarray:
        b = np.sum(a, axis=0, dtype="float")
        f = b == 0
        y = np.array(a, copy=True)
        y[:, f] = 1
        b[f] = y.shape[0]
        return y / b

    if axis == 0:
        return handle_zero_rows(X)
    elif axis == 1:
        return handle_zero_rows(X.T).T
    else:
        return handle_all_zeros(X)


def dispersion(w: np.ndarray) -> float:
    return -np.sum(w[w > 0.0] * np.log(w[w > 0.0]))  # filter 0 as 0 * -inf is undef in NumPy


def ndispersion(w: np.ndarray) -> float:
    return dispersion(w) / np.log(len(w))


def yager_orness(w: np.ndarray) -> float:
    """
    The orness is a measure of how "or-like" a given weight vector is for use in OWA.

    orness(w) = 1/(n-1) * sum( (n-i)*w )
    """
    n = len(w)
    return np.sum(np.arange(n - 1, -1, -1) * w) / (n - 1.0)


def yager_andness(w: np.ndarray) -> float:
    """
    Yager's andness is 1.0 - Yager's orness for a given weight vector.
    """
    return 1.0 - yager_orness(w)


def weights_mapping(w: np.ndarray) -> np.ndarray:
    s = np.e**w
    return s / np.sum(s)


class OWA:
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

    def __init__(self, v: np.ndarray):
        self.v = v
        self.v_ = v[::-1]  # save the inverse so we don't need to reverse np.sort
        self.lv = len(v)

    def __call__(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        if X.shape[axis] != self.lv:
            raise ValueError("len(X) != len(v)")
        b = np.sort(X, axis)  # creates permutation
        return self.sorted_mean(b, axis)

    def sorted_mean(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        """Use for pre-sorted X"""
        # align the weight vector along the aggregated axis
        shape = [1] * X.ndim
        shape[axis] = self.lv
        return np.sum(X * self.v_.reshape(shape), axis)

    def __str__(self) -> str:
        return "OWA(" + " ".join([f"{x:.4f}" for x in self.v]) + ")"

    def __repr__(self) -> str:
        return str(self)

    def andness(self) -> float:
        return yager_andness(self.v)

    def orness(self) -> float:
        return yager_orness(self.v)

    def disp(self) -> float:
        return dispersion(self.v)

    def ndisp(self) -> float:
        return ndispersion(self.v)


class GOWA(OWA):
    """
    Generalized order weighted averaging operator.

    The generalized order weighted averaging operator aggregates a vector of a1, ..., an using
    a permutation b1, ..., bn for which b1 >= b2 => ... => bn where b1 is the largest value in
    a, and which has an related weight vector w = w1, ..., wn in [0, 1] and sum w = 1.

    Averaging is used using the power-mean: sum(w*b^p)^(1/p), where p is the power parameter.
    """

    def __init__(self, p: float, v: np.ndarray):
        """
        Constructs GOWA operator.

        Parameters:
        -----------
        p : power parameter.

        v : weights.
        """
        super().__init__(v)
        self.p = p
        self.inv_p = 1.0 / p

    def sorted_mean(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        shape = [1] * X.ndim
        shape[axis] = self.lv
        return np.sum((X**self.p) * self.v_.reshape(shape), axis) ** self.inv_p

    def __str__(self) -> str:
        return (f"GOWA({self.p:f}, ") + " ".join([f"{x:.4f}" for x in self.v]) + ")"


def gowa(p: float, *w: Any) -> GOWA:
    """Create Generalized OWA (GOWA) operator from weights"""
    w_arr = np.asarray(w).ravel()
    return GOWA(p, w_arr)


def owa(*w: Any) -> OWA:
    """Create OWA operator from weights"""
    w_arr = np.asarray(w).ravel()
    return OWA(w_arr)


def meowa(n: int, orness: float, **kwargs: Any) -> OWA:
    """
    Maximize dispersion at a specified orness level.

    This method uses O'Hagan's method for finding the MEOWA weights.
    """
    if 0.0 > orness or orness > 1.0:
        raise ValueError("orness must be in [0, 1]")

    if n < 2:
        raise ValueError("n must be > 1")

    # edge cases
    if np.isclose(orness, 0.5):
        return OWA(np.ones(n) / n)
    elif np.isclose(orness, 0.0):
        w = np.zeros(n)
        w[-1] = 1.0
        return OWA(w)
    elif np.isclose(orness, 1.0):
        w = np.zeros(n)
        w[0] = 1.0
        return OWA(w)

    try:
        from scipy.optimize import root_scalar
    except ImportError:
        raise ImportError(
            "The 'scipy' library is required for this functionality. Please install it with `pip install scipy`."
        )

    # helper to calculate weights from h
    def get_w(h: float) -> np.ndarray:
        # w_i = h^{n-i} / sum(h^{n-j})
        p = np.power(h, np.arange(n - 1, -1, -1))
        return p / np.sum(p)

    # function to find root for
    def f(h: float) -> float:
        return yager_orness(get_w(h)) - orness

    if orness < 0.5:
        # h in [0, 1]
        bracket = [0.00001, 0.99999]
    else:
        # h in [1, inf]
        # heuristic upper bound
        ub = 2.0
        while f(ub) < 0:
            ub *= 2.0
            if ub > 1e10:  # safety break
                raise ValueError(f"Could not bound root for orness {orness:f}")
        bracket = [1.00001, ub]

    # filter kwargs for root_scalar
    rs_args = {k: v for k, v in kwargs.items() if k in ["maxiter", "xtol", "rtol"]}

    res = root_scalar(f, bracket=bracket, method="brentq", **rs_args)

    if res.converged:
        return OWA(get_w(res.root))
    else:
        raise ValueError("Could not optimize weights: " + str(res))


def sampling_owa_orness(x: np.ndarray, d: float, **kwargs: Any) -> OWA:
    """
    Maximize orness of an owa operator given a given result data point.
    """
    n = len(x)
    if n < 2:
        raise ValueError("n must be > 1")

    s_ = np.sort(x)[::-1]

    def negorness(v: np.ndarray) -> float:
        return -yager_orness(v)

    def constraint_has_output_d(v: np.ndarray) -> float:
        return np.sum(s_ * v) - d

    def constraint_has_sum(v: np.ndarray) -> float:
        return np.sum(v) - 1.0

    return _minimize_owa(negorness, (constraint_has_sum, constraint_has_output_d), n, **kwargs)


def sampling_owa_ndisp(x: np.ndarray, d: float, **kwargs: Any) -> OWA:
    """
    Maximize dispersion of an owa operator given a given result data point.
    """
    n = len(x)
    if n < 2:
        raise ValueError("n must be > 1")

    s_ = np.sort(x)[::-1]

    def negndisp(v: np.ndarray) -> float:
        return -ndispersion(v)

    def constraint_has_output_d(v: np.ndarray) -> float:
        return np.sum(s_ * v) - d

    def constraint_has_sum(v: np.ndarray) -> float:
        return np.sum(v) - 1.0

    return _minimize_owa(negndisp, (constraint_has_sum, constraint_has_output_d), n, **kwargs)


def mvowa(n: int, orness: float, **kwargs: Any) -> OWA:
    """
    Maximum variability order weighted aggregation. Construct aggregation with fixed orness
    but maximized variance. [Fuller and Majlender, 2003]
    """
    if 0.0 > orness or orness > 1.0:
        raise ValueError("orness must be in [0, 1]")

    if n < 2:
        raise ValueError("n must be > 1")

    def variance(v: np.ndarray) -> float:
        return np.var(v)

    def constraint_has_orness(v: np.ndarray) -> float:
        return yager_orness(v) - orness

    def constraint_has_sum(v: np.ndarray) -> float:
        return np.sum(v) - 1.0

    return _minimize_owa(variance, (constraint_has_orness, constraint_has_sum), n, **kwargs)


def _minimize_owa(
    minfunc: Callable[[np.ndarray], float],
    constraints: Sequence[Callable[[np.ndarray], float]],
    n: int,
    **kwargs: Any
) -> OWA:
    try:
        from scipy.optimize import minimize
    except ImportError:
        raise ImportError(
            "The 'scipy' library is required for this functionality. Please install it with `pip install scipy`."
        )

    bounds = tuple([(0, 1) for x in range(n)])  # this is actually the third constraint, but common.

    initial = np.ones(n) / n

    constraints_ = tuple([{"fun": c, "type": "eq"} for c in constraints])

    res = minimize(minfunc, initial, bounds=bounds, options=kwargs, constraints=constraints_)

    if res.success:
        return OWA(res.x)
    else:
        raise ValueError("Could not optimize weights: " + res.message)


class AndnessDirectedAveraging:
    def __init__(self, p: float):
        self.p = p
        self.tnorm = p <= 0.5
        self.alpha = (1.0 - p) / p if self.tnorm else p / (1.0 - p)

    def __call__(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        X = np.asarray(X)
        if self.tnorm:
            return (np.sum(X**self.alpha, axis) / X.shape[axis]) ** (1.0 / self.alpha)
        else:
            return 1.0 - ((np.sum((1.0 - X) ** (1.0 / self.alpha), axis) / X.shape[axis]) ** self.alpha)


def aa(p: float) -> AndnessDirectedAveraging:
    assert 0 < p and p < 1
    return AndnessDirectedAveraging(p)
