"""Tests for fylearn.canonicalize (tree canonicalization identities)."""

from __future__ import annotations

import numpy as np
import pytest

from fylearn import fuzzylogic as fl
from fylearn.canonicalize import (
    ConstLeaf,
    canonical_form,
    canonical_key,
    functional_signature,
    functionally_equivalent,
    serialize,
    simplify,
)
from fylearn.fpt import Inner, Leaf

X_GRID = np.column_stack([np.linspace(0, 1, 101)] * 2)


def make_leaf(idx: int, name: str, a: float, b: float, c: float) -> Leaf:
    return Leaf(idx, name, fl.TriangularSet(a, b, c))


H0 = make_leaf(0, "hig", 0.5, 1.0, 1.0)
M0 = make_leaf(0, "med", 0.0, 0.5, 1.0)
L0 = make_leaf(0, "low", 0.0, 0.0, 0.5)
H1 = make_leaf(1, "hig", 0.5, 1.0, 1.0)
L1 = make_leaf(1, "low", 0.0, 0.0, 0.5)


def inner(op, a, b):
    return Inner(op, [a, b])


class TestCommutativeSorting:
    def test_min_reorder(self):
        t1 = inner(fl.min, H0, L1)
        t2 = inner(fl.min, L1, H0)
        assert canonical_key(t1) == canonical_key(t2)

    def test_all_restricted_ops_reorder(self):
        for op in (fl.prod, fl.mean, fl.max):
            assert canonical_key(inner(op, H0, L1)) == canonical_key(inner(op, L1, H0))

    def test_nested_reorder(self):
        t1 = inner(fl.max, M0, inner(fl.prod, H0, L1))
        t2 = inner(fl.max, inner(fl.prod, L1, H0), M0)
        assert canonical_key(t1) == canonical_key(t2)

    def test_serialize_deterministic(self):
        t = inner(fl.min, inner(fl.mean, H0, L1), M0)
        assert serialize(t) == serialize(t)


class TestSimplification:
    @pytest.mark.parametrize(("op", "neutral"), [(fl.min, 1.0), (fl.prod, 1.0)])
    def test_neutral_removal_min_prod(self, op, neutral):
        assert canonical_key(inner(op, H0, ConstLeaf(neutral))) == canonical_key(H0)

    def test_neutral_removal_max(self):
        assert canonical_key(inner(fl.max, H0, ConstLeaf(0.0))) == canonical_key(H0)

    @pytest.mark.parametrize(("op", "zero"), [(fl.min, 0.0), (fl.prod, 0.0)])
    def test_absorbing_zero_min_prod(self, op, zero):
        s = simplify(inner(op, H0, ConstLeaf(zero)))
        assert isinstance(s, ConstLeaf) and s.value == 0.0

    def test_absorbing_one_max(self):
        s = simplify(inner(fl.max, H0, ConstLeaf(1.0)))
        assert isinstance(s, ConstLeaf) and s.value == 1.0

    @pytest.mark.parametrize("op", [fl.min, fl.max, fl.mean])
    def test_duplicate_collapse(self, op):
        assert canonical_key(inner(op, H0, H0)) == canonical_key(H0)

    def test_prod_duplicates_preserved(self):
        assert canonical_key(inner(fl.prod, H0, H0)) != canonical_key(H0)

    def test_single_child_collapse(self):
        assert canonical_key(Inner(fl.min, [H0])) == canonical_key(H0)

    def test_constant_folding(self):
        s = simplify(inner(fl.min, ConstLeaf(0.3), ConstLeaf(0.7)))
        assert isinstance(s, ConstLeaf) and np.isclose(s.value, 0.3)


class TestRounding:
    def test_params_rounded(self):
        a = Leaf(0, "x", fl.TriangularSet(0.300004, 0.500004, 1.0))
        b = Leaf(0, "x", fl.TriangularSet(0.299996, 0.499996, 1.0))
        assert canonical_key(a, 1e-4) == canonical_key(b, 1e-4)
        assert canonical_key(a, 1e-6) != canonical_key(b, 1e-6)


class TestIdempotence:
    @pytest.mark.parametrize(
        "tree",
        [
            lambda: H0,
            lambda: inner(fl.min, H0, L1),
            lambda: inner(fl.max, M0, inner(fl.prod, H0, L1)),
            lambda: inner(fl.min, H0, ConstLeaf(1.0)),
            lambda: inner(fl.prod, H0, ConstLeaf(0.0)),
        ],
    )
    def test_canonical_form_idempotent(self, tree):
        t = tree()
        assert canonical_key(canonical_form(t)) == canonical_key(t)


class TestFunctionalEquivalence:
    def test_commutative_pair_equivalent(self):
        assert functionally_equivalent(inner(fl.min, H0, L1), inner(fl.min, L1, H0), X_GRID)

    def test_lattice_absorption(self):
        # max(h0, min(h0, l1)) == h0 pointwise
        t2 = inner(fl.max, H0, inner(fl.min, H0, L1))
        assert functionally_equivalent(H0, t2, X_GRID, tolerance=1e-6)

    def test_different_functions_not_equivalent(self):
        assert not functionally_equivalent(H0, inner(fl.max, H0, L1), X_GRID)

    def test_signature_shape_and_range(self):
        sig = functional_signature(inner(fl.mean, H0, L1), X_GRID)
        assert sig.shape == (101,)
        assert 0.0 <= sig.min() and sig.max() <= 1.0
