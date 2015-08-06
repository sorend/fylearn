# -*- coding: utf-8 -*-
"""Fuzzy pattern tree based methods

The module structure is the following:

- The "FuzzyPatternTreeClassifier" implements the fit logic for bottom-up
  construction of the fuzzy pattern tree [1].

- The "FuzzyPatternTreeTopDownClassifier" implements the fit logic for top-down
  construction of the fuzzy pattern tree [2].

- The "FuzzyPatternTreeRegressor" implements a regressor based on
  top-down constructed fuzzy pattern tree [3].

References:
[1] Hwong, 2009.
[2] Senge and Huellemeier, 2009.
[3] Senge and Huellemeier, 2010.
"""

import numpy as np
import heapq
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
import fylearn.fuzzylogic as fl

# aggregation operators to use
OPERATORS = (
    fl.min,
    fl.einstein_i,
    fl.lukasiewicz_i,
    fl.prod,
    fl.owa([0.2, 0.8]),
    fl.owa([0.4, 0.6]),
    fl.mean,
    fl.owa([0.6, 0.4]),
    fl.owa([0.8, 0.2]),
    fl.algebraic_sum,
    fl.lukasiewicz_u,
    fl.einstein_u,
    fl.max
)

def _tree_iterator(root):
    Q = [ root ]
    while Q:
        tree = Q.pop(0)
        if isinstance(tree, Inner):
            Q.extend(tree.branches_)
        yield tree

def _tree_leaves(root):
    return [ x for x in _tree_iterator(root) if isinstance(x, Leaf) ]

def _tree_clone_replace_leaf(root, replace_node, new_node):
    if root == replace_node:
        return new_node
    else:
        if isinstance(root, Leaf):
            return root
        else:
            new_branches = [ _tree_clone_replace_leaf(b, replace_node, new_node) for b in root.branches_ ]
            return Inner(root.aggregation_, new_branches)

def _tree_contains(root, to_find):
    for n in _tree_iterator(root):
        if n == to_find:
            return True
    return False

def default_rmse(a, b):
    return 1.0 - mean_squared_error(a, b)

def default_fuzzifier(idx, F):
    # get min/max from data
    v_min = np.nanmin(F)
    v_max = np.nanmax(F)
    # blarg
    return [ Leaf(idx, "low", fl.TriangularSet(v_min - (v_max - v_min) ** 2, v_min, v_max)),
             Leaf(idx, "med", fl.TriangularSet(v_min, v_min + ((v_max - v_min) / 2), v_max)),
             Leaf(idx, "hig", fl.TriangularSet(v_min, v_max, v_max + (v_max - v_min) ** 2)) ]

def _select_candidates(candidates, n_select, class_vector, similarity_measure, X):
    """Select a number of candidate trees with the best similarity to the class vector."""
    R = [ _evaluate_similarity(c, class_vector, similarity_measure, X) for c in candidates ]
    return heapq.nlargest(n_select, R, key=lambda x: x[0])

def _evaluate_similarity(candidate, class_vector, similarity_measure, X):
    y_pred = candidate(X)
    s = similarity_measure(y_pred, class_vector)
    return (s, candidate)

class Tree:
    pass

class Leaf(Tree):
    """Leaf node in the tree, contains index of the feature and the membership function to apply"""
    def __init__(self, idx, name, mu):
        self.idx = idx
        self.name = name
        self.mu = mu

    def __repr__(self):
        return "Leaf(" + repr(self.idx) + "_" + self.name + ")"

    def __call__(self, X):
        return self.mu(X[:, self.idx])  # apply the membership function to the specific feature idx

class Inner(Tree):
    """Branching node in the tree """
    def __init__(self, aggregation, branches):
        self.branches_ = branches
        self.aggregation_ = aggregation

    def __repr__(self):
        return "(" + repr(self.aggregation_.__name__) + ", " + ", ".join([ repr(x) for x in self.branches_ ]) + ")"

    def __call__(self, X):
        # output for each branches
        R = np.zeros((X.shape[0], len(self.branches_)))
        for idx, branch in enumerate(self.branches_):
            R[:, idx] = branch(X)
        return self.aggregation_(R)

class FuzzyPatternTreeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 similarity_measure=default_rmse,
                 max_depth=5,
                 num_candidates=2,
                 num_slaves=3,
                 fuzzifier=default_fuzzifier):
        self.similarity_measure = similarity_measure
        self.max_depth = max_depth
        self.num_candidates = num_candidates
        self.num_slaves = num_slaves
        self.fuzzifier = fuzzifier

    def get_params(self, deep=True):
        return {"similarity_measure": self.similarity_measure,
                "max_depth": self.max_depth,
                "num_candidates": self.num_candidates,
                "num_slaves": self.num_slaves,
                "fuzzifier": self.fuzzifier}

    def set_params(self, **params):
        for key, value in params.items():
            self.setattr(key, value)
        return self

    def fit(self, X, y):

        X = check_array(X)

        self.classes_, y = np.unique(y, return_inverse=True)

        if np.nan in self.classes_:
            raise "nan not supported for class values"

        self.trees_ = {}

        # build membership functions
        P = []
        for feature_idx, feature in enumerate(X.T):
            P.extend(self.fuzzifier(feature_idx, feature))

        # build the pattern tree for each class
        for class_idx, class_value in enumerate(self.classes_):
            class_vector = np.zeros(len(y))
            class_vector[y == class_idx] = 1.0
            root = self.build_for_class(X, y, class_vector, list(P))
            self.trees_[class_idx] = root

        return self

    def build_for_class(self, X, y, class_vector, P):
        S = []
        C = _select_candidates(P, self.num_candidates, class_vector, self.similarity_measure, X)

        for depth in range(self.max_depth):
            P_U_S = list(P)
            P_U_S.extend([ s[1] for s in S ])

            new_candidates = self.select_slaves(C, P_U_S, class_vector, X)

            # no new candidates found
            if len(new_candidates) == 0:
                break

            S.extend(new_candidates)

            # no better similarity received
            if new_candidates[0][0] < C[0][0]:
                break

            # clean out primitive trees
            for s in S:
                P = [ p for p in P if not _tree_contains(s[1], p) ]

            # remove primitives already in candidates
            for c in new_candidates:
                P = [ p for p in P if not _tree_contains(c[1], p) ]

            C = new_candidates

        # first candidates
        return C[0][1]

    def select_slaves(self, candidates, P_U_S, class_vector, X):
        R = []
        for candidate in candidates:
            aggregates = []
            for other in P_U_S:
                if not _tree_contains(candidate[1], other):
                    aggregates.extend([ Inner(a, [ candidate[1], other ]) for a in OPERATORS ])

            R.extend(_select_candidates(aggregates, self.num_slaves, class_vector, self.similarity_measure, X))

        R = sorted(R, key=lambda x: x[0])

        RR = []
        used_nodes = set()
        for candidate in R:
            inner_node = candidate[1]
            found = False
            for tree in inner_node.branches_:
                if tree in used_nodes:
                    found = True
            if not found:
                used_nodes.update(inner_node.branches_)
                RR.append(candidate)

        return heapq.nlargest(self.num_slaves, RR, key=lambda x: x[0])

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : Array-like of shape [n_samples, n_features]
            The input to classify.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """

        X = check_array(X)

        if self.trees_ is None:
            raise Exception("Pattern trees not initialized. Perform a fit first.")

        y_classes = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            y_classes[:, i] = self.trees_[i](X)

        # predict the maximum value
        return self.classes_.take(np.argmax(y_classes, -1))


class FuzzyPatternTreeTopDownClassifier(FuzzyPatternTreeClassifier):
    """
    Fuzzy Pattern Tree with Top Down induction algorithm.
    """

    def __init__(self,
                 similarity_measure=default_rmse,
                 relative_improvement=0.01,
                 num_candidates=5,
                 fuzzifier=default_fuzzifier):
        self.similarity_measure = similarity_measure
        self.relative_improvement = relative_improvement
        self.num_candidates = num_candidates
        self.fuzzifier = fuzzifier

    def get_params(self, deep=True):
        return {"similarity_measure": self.similarity_measure,
                "relative_improvement": self.relative_improvement,
                "num_candidates": self.num_candidates,
                "fuzzifier": self.fuzzifier}

    def select_slaves(self, C, P, class_vector, num_candidates, X):

        R = []
        for candidate in C:
            c = candidate[1]
            modified = []
            candidate_leaves = _tree_leaves(c)

            for c_leaf in candidate_leaves:
                for p_leaf in [ p for p in P if p not in candidate_leaves ]:
                    for aggr in OPERATORS:
                        modified.append(_tree_clone_replace_leaf(c, c_leaf, Inner(aggr, [ c_leaf, p_leaf ])))

            R.extend(_select_candidates(modified, self.num_candidates, class_vector, self.similarity_measure, X))

            R = list(heapq.nlargest(self.num_candidates, R, key=lambda x: x[0]))

        return list(reversed(sorted(R, key=lambda x: x[0])))

    def build_for_class(self, X, y, class_vector, P):

        C = _select_candidates(P, self.num_candidates, class_vector, self.similarity_measure, X)
        C = sorted(C, key=lambda x: x[0])

        while True:

            if C[0][0] == 1.0:
                break

            new_candidates = self.select_slaves(C, P, class_vector, self.num_candidates, X)

            if len(new_candidates) == 0:
                break

            if new_candidates[0][0] < (1.0 + self.relative_improvement) * C[0][0]:
                break

            C = new_candidates

        return C[0][1]
