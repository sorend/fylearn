"""Canonicalization of fuzzy pattern trees.

Deterministic serialization, algebraic simplification, and functional
equivalence checking for the trees built by :mod:`fylearn.fpt`
(:class:`fylearn.fpt.Leaf` and :class:`fylearn.fpt.Inner`).

The purpose is to support Rashomon-set style analysis: syntactically different
encodings can describe the same fuzzy function, so before counting models each
representation is canonicalized by

- sorting children of commutative operators;
- collapsing duplicate subtrees for idempotent operators (min, max, mean, OWA);
- removing neutral constants (1 for min/prod, 0 for max) and propagating
  absorbing constants (0 for min/prod, 1 for max);
- folding constant subtrees and collapsing single-child inner nodes;
- rounding membership parameters to a declared resolution.

Functional equivalence is checked separately on a dense reference grid.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from fylearn import fuzzylogic as fl
from fylearn.fpt import Inner, Leaf, Tree, TreeEvaluator

RESTRICTED_OPERATOR_NAMES: dict[Callable, str] = {
    fl.min: "min",
    fl.prod: "prod",
    fl.mean: "mean",
    fl.max: "max",
}

_IDEMPOTENT_DROP_OPS: frozenset[str] = frozenset({"min", "max", "mean", "owa", "gowa"})

_NEUTRAL: dict[str, float] = {"min": 1.0, "prod": 1.0, "max": 0.0}
_ABSORBING: dict[str, float] = {"min": 0.0, "prod": 0.0, "max": 1.0}


# ---------------------------------------------------------------------------
# operator identity
# ---------------------------------------------------------------------------


def op_name(aggregation: Callable) -> str:
    """Canonical operator name (restricted library or function name / repr)."""
    name = RESTRICTED_OPERATOR_NAMES.get(aggregation)
    if name is not None:
        return name
    name = getattr(aggregation, "__name__", None)
    if name:
        return name
    return repr(aggregation)


def is_commutative(aggregation: Callable) -> bool:
    name = op_name(aggregation)
    if name in RESTRICTED_OPERATOR_NAMES.values():
        return True
    return name.startswith("owa") or name.startswith("OWA") or name.startswith("GOWA")


# ---------------------------------------------------------------------------
# membership parameters
# ---------------------------------------------------------------------------


def _fmt_param(value: float, resolution: float | None) -> str:
    if resolution is None or resolution <= 0:
        return repr(float(value))
    ndec = max(0, -int(math.floor(math.log10(resolution))))
    return f"{value:.{ndec}f}"


def mu_params(mu: Any, resolution: float | None = 1e-4) -> tuple[Any, ...]:
    """Extract rounded membership parameters as a tuple."""
    if isinstance(mu, fl.TriangularSet):
        return tuple(_fmt_param(v, resolution) for v in (mu.a, mu.b, mu.c))
    if isinstance(mu, fl.TrapezoidalSet):
        return tuple(_fmt_param(v, resolution) for v in (mu.a, mu.b, mu.c, mu.d))
    if isinstance(mu, fl.PiSet):
        return tuple(_fmt_param(v, resolution) for v in (mu.a, mu.r, mu.b, mu.m))
    if isinstance(mu, fl.ZadehNegatedSet):
        return ("not",) + mu_params(mu.s, resolution)
    if type(mu).__name__ == "CrispSet":
        return (_fmt_param(mu.value, resolution),)
    return (str(mu),)


def leaf_label(leaf: Leaf, resolution: float | None = 1e-4) -> str:
    """Canonical label for a leaf: feature index, predicate name, rounded params."""
    params = mu_params(leaf.mu, resolution)
    return f"L{leaf.idx}_{leaf.name}[{','.join(params)}]"


# ---------------------------------------------------------------------------
# constant leaves (for simplification)
# ---------------------------------------------------------------------------


class ConstLeaf(Tree):
    """Constant-valued leaf used during simplification."""

    __slots__ = ["value"]

    def __init__(self, value: float):
        self.value = float(value)

    def __repr__(self) -> str:
        return f"ConstLeaf({self.value:g})"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.full((X.shape[0],), self.value)


def _apply_op(aggregation: Callable, vals: np.ndarray) -> float:
    name = op_name(aggregation)
    if name == "min":
        return float(np.min(vals))
    if name == "max":
        return float(np.max(vals))
    if name == "prod":
        return float(np.prod(vals))
    if name == "mean":
        return float(np.mean(vals))
    try:
        return float(aggregation(vals))
    except Exception as exc:
        raise ValueError(f"cannot fold operator {name} over constants: {exc}") from exc


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def serialize(tree: Tree, resolution: float | None = 1e-4) -> str:
    """Deterministic serialization with commutative children sorted (no simplification)."""
    if isinstance(tree, ConstLeaf):
        return f"C{tree.value:g}"
    if isinstance(tree, Leaf):
        return leaf_label(tree, resolution)
    if isinstance(tree, Inner):
        children = [serialize(b, resolution) for b in tree.branches_]
        if is_commutative(tree.aggregation_):
            children = sorted(children)
        return f"I({op_name(tree.aggregation_)})[{';'.join(children)}]"
    raise TypeError(f"unsupported node type: {type(tree)}")


# ---------------------------------------------------------------------------
# simplification
# ---------------------------------------------------------------------------


def simplify(tree: Tree, resolution: float | None = 1e-4) -> Tree:
    """Algebraically simplify a tree for canonicalization."""
    if isinstance(tree, (ConstLeaf, Leaf)):
        return tree
    if isinstance(tree, Inner):
        children = [simplify(b, resolution) for b in tree.branches_]
        name = op_name(tree.aggregation_)

        if children and all(isinstance(c, ConstLeaf) for c in children):
            vals = np.array([c.value for c in children])
            return ConstLeaf(_apply_op(tree.aggregation_, vals))

        if name in _NEUTRAL:
            neutral = _NEUTRAL[name]
            children = [c for c in children if not (isinstance(c, ConstLeaf) and math.isclose(c.value, neutral))]
        if name in _ABSORBING:
            absorbing = _ABSORBING[name]
            if any(isinstance(c, ConstLeaf) and math.isclose(c.value, absorbing) for c in children):
                return ConstLeaf(absorbing)
        if name in _IDEMPOTENT_DROP_OPS:
            seen: dict[str, Tree] = {}
            for c in children:
                seen.setdefault(canonical_key(c, resolution), c)
            children = list(seen.values())

        if len(children) == 1:
            return children[0]
        if not children:
            return tree

        if is_commutative(tree.aggregation_):
            children = sorted(children, key=lambda c: canonical_key(c, resolution))
        return Inner(tree.aggregation_, children)
    raise TypeError(f"unsupported node type: {type(tree)}")


def canonical_key(tree: Tree, resolution: float | None = 1e-4) -> str:
    """Canonical structural key: serialize(simplify(tree))."""
    return serialize(simplify(tree, resolution), resolution)


def canonical_form(tree: Tree, resolution: float | None = 1e-4) -> Tree:
    """Return the simplified canonical tree object."""
    return simplify(tree, resolution)


# ---------------------------------------------------------------------------
# functional equivalence
# ---------------------------------------------------------------------------


def functional_signature(tree: Tree, X: np.ndarray, resolution: float | None = 1e-3) -> np.ndarray:
    """Membership values of the tree on reference inputs, rounded to resolution."""
    values = TreeEvaluator(np.asarray(X)).predict(tree)
    if resolution is None or resolution <= 0:
        return values
    return np.round(values / resolution) * resolution


def functional_key(tree: Tree, X: np.ndarray, resolution: float | None = 1e-3) -> bytes:
    return functional_signature(tree, X, resolution).astype(np.float64).tobytes()


def functionally_equivalent(t1: Tree, t2: Tree, X: np.ndarray, tolerance: float = 1e-3) -> bool:
    s1 = functional_signature(t1, X, None)
    s2 = functional_signature(t2, X, None)
    return bool(np.max(np.abs(s1 - s2)) <= tolerance)


def functional_classes(
    trees: Sequence[Tree], X: np.ndarray, resolution: float | None = 1e-3
) -> dict[bytes, list[int]]:
    """Group tree indices into functional equivalence classes by rounded signatures."""
    classes: dict[bytes, list[int]] = {}
    for i, t in enumerate(trees):
        classes.setdefault(functional_key(t, X, resolution), []).append(i)
    return classes
