# -*- coding: utf-8 -*-
"""
Fuzzy Pattern Classifiers.


References:
-----------
[1] Davidsen 2015.

[2] Monks, 2009.

[3] Davidsen, 2016.

[4] Meher, 2009.

"""

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.metrics import mean_squared_error
from .fuzzylogic import PiSet, TriangularSet, owa, meowa, p_normalize, prod, weights_mapping
from .ga import UnitIntervalGeneticAlgorithm, helper_fitness, helper_n_generations
from .tlbo import TLBO
from .local_search import PatternSearchOptimizer, helper_generations, LocalUnimodalSamplingOptimizer

logger = logging.getLogger(__name__)

def pi_factory(**kwargs):
    m = kwargs["m"] if "m" in kwargs else 2.0
    c = kwargs["mean"]
    d = (kwargs["max"] - kwargs["min"]) / 2.0
    return PiSet(a=c - d, r=c, b=c + d, m=m)

def t_factory(**kwargs):
    c = kwargs["mean"]
    d = (kwargs["max"] - kwargs["min"]) / 2.0
    return TriangularSet(c - d, c, c + d)

def build_memberships(X, class_idx, factory):
    # take column-wise min/mean/max for class
    mins = np.nanmin(X[class_idx], 0)
    means = np.nanmean(X[class_idx], 0)
    maxs = np.nanmax(X[class_idx], 0)
    return [ factory(min=mins[i], mean=means[i], max=maxs[i]) for i in range(X.shape[1]) ]


def learn_class(X, y, class_idx, membership_factory, aggregation_factory):
    mus = build_memberships(X, class_idx, membership_factory)
    aggr = aggregation_factory(mus, X, y, class_idx)
    return mus, aggr

#
# Authors: Søren Atmakuri Davidsen <sorend@gmail.com>
#

def predict_proto(X, proto, aggregation, A):
    for col_no in range(X.shape[1]):
        A[:, col_no] = proto[col_no](X[:, col_no])
    return aggregation(A, axis=1)

def predict_protos(X, protos, aggregation):
    y = np.zeros((X.shape[0], len(protos)))
    A = np.zeros(X.shape)  # re-use this matrix
    for clz_no, proto in enumerate(protos):
        y[:, clz_no] = predict_proto(X, proto, aggregation, A)
    return y

def predict_protos_aggregations(X, protos, aggregations):
    y = np.zeros((X.shape[0], len(protos)))
    A = np.zeros(X.shape)  # re-use this matrix
    for clz_no, proto in enumerate(protos):
        y[:, clz_no] = predict_proto(X, proto, aggregations[clz_no], A)
    return y

class generations_optimizer(object):
    """Will optimize OWA for any generations based optimizer"""
    def __init__(self, s, of):
        self.s = s
        self.of = of

    def __call__(self, X, fitness):
        iterations, o = self.of(X, fitness)
        o = helper_n_generations(o, iterations)
        chromosomes, fitnesses = o.best(1)
        return chromosomes[0]

    def __str__(self):
        return self.s

def ga_owa_optimizer(f_evals=5):
    """ GA OWA optimizer """
    def factory(X, fitness):
        return f_evals * X.shape[1], UnitIntervalGeneticAlgorithm(
            fitness_function=helper_fitness(fitness),
            n_chromosomes=50,
            elitism=3,
            p_mutation=0.1,
            n_genes=X.shape[1])
    return generations_optimizer("ga", factory)

def tlbo_owa_optimizer(f_evals=5):
    """TLBO OWA optimizer"""
    def factory(X, fitness):
        return f_evals * X.shape[1], TLBO(
            f=fitness, lower_bound=np.zeros(X.shape[1]), upper_bound=np.zeros(X.shape[1]))
    return generations_optimizer("tlbo", factory)

def ps_owa_optimizer(f_evals=5):
    """Pattern Search OWA optimizer"""
    def factory(X, fitness):
        return 10, helper_generations(PatternSearchOptimizer(
            fitness,
            np.zeros(X.shape[1]), np.ones(X.shape[1]),
            max_evaluations=X.shape[1] * f_evals))
    return generations_optimizer("ps", factory)

def lus_owa_optimizer(f_evals=10):
    """Local Unimodal Sampling OWA optimizer"""
    def factory(X, fitness):
        return 10, helper_generations(LocalUnimodalSamplingOptimizer(
            fitness,
            np.zeros(X.shape[1]), np.ones(X.shape[1]),
            max_evaluations=X.shape[1] * f_evals))
    return generations_optimizer("lus", factory)

def build_y_target(y, classes):
    y_target = np.zeros((len(y), len(classes)))
    for i, c in enumerate(classes):
        y_target[y == i, i] = 1.0
    return y_target

def evaluate_rmse(y_target, y_pred):
    if np.isnan(np.sum(y_pred)):
        return 1.0
    else:
        return mean_squared_error(y_target, y_pred)

def owa_decoder_plain(c):
    return owa(weights_mapping(c))

class GAOWAFactory:

    def __init__(self, optimizer=ga_owa_optimizer(), decoder=owa_decoder_plain):
        self.optimizer = optimizer
        self.decoder = decoder

    def __call__(self, protos, X, y, classes):

        y_target = build_y_target(y, classes)

        def fitness(c):
            aggr = self.decoder(c)
            y_pred = predict_protos(X, protos, aggr)
            return evaluate_rmse(y_target, y_pred)

        weights = self.optimizer(X, fitness)

        best = self.decoder(weights)

        logger.info("trained owa(%s, %s, %s)" % (str(self.optimizer),
                                                 str(self.decoder).split(" ")[1].split("_")[-1],
                                                 ", ".join(map(lambda x: "%.5f" % (x,), best.v))))

        return best

class StaticFactory:

    def __init__(self, aggregation=prod):
        self.aggregation = aggregation

    def __call__(self, *args, **kwargs):
        return self.aggregation

class MEOWAFactory:

    def __call__(self, protos, X, y, classes):

        y_target = build_y_target(y, classes)

        def fitness(orness):
            aggr = meowa(X.shape[1], orness[0], maxiter=1000)
            y_pred = predict_protos(X, protos, aggr)
            return evaluate_rmse(y_target, y_pred)

        lower_bounds = (0.0,)
        upper_bounds = (1.0,)

        ps = helper_generations(PatternSearchOptimizer(fitness, lower_bounds, upper_bounds, max_evaluations=5))
        ps = helper_n_generations(ps, 10)
        best_orness, best_fit = ps.best(1)

        best = meowa(X.shape[1], best_orness[0][0], maxiter=1000)  # construct from optimizer

        logger.info("trained owa(meowa, plain, %s)" % (", ".join(map(lambda x: "%.5f" % (x,), best.v))))

        return best

class FuzzyPatternClassifier(BaseEstimator, ClassifierMixin):
    """
    Fuzzy pattern classifier using aggregation factory and membership factory.
    """
    def get_params(self, deep=False):
        return {"aggregation_factory": self.aggregation_factory,
                "membership_factory": self.membership_factory}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def __init__(self, membership_factory=pi_factory, aggregation_factory=StaticFactory(prod)):
        """
        Constructs object

        Parameters:
        -----------

        membership_factory: the function to use for creating membership function.

        aggregation_factory: the function to use for creating the aggregation function.

        """
        self.aggregation_factory = aggregation_factory
        self.membership_factory = membership_factory

    def fit(self, X, y):

        X = check_array(X)

        self.classes_, y = np.unique(y, return_inverse=True)

        if "?" in tuple(self.classes_):
            raise ValueError("nan not supported for class values")

        # build membership functions for each feature for each class
        self.protos_ = [
            build_memberships(X, y == idx, self.membership_factory)
            for idx, class_value in enumerate(self.classes_)
        ]

        # build aggregation
        self.aggregation_ = self.aggregation_factory(self.protos_, X, y, self.classes_)

        return self

    def predict(self, X):
        """
        Predicts if examples in X belong to classifier's class or not.

        Parameters
        ----------
        X : examples to predict for.
        """
        if not hasattr(self, "classes_"):
            raise Exception("Perform a fit first.")

        y_mu = predict_protos(X, self.protos_, self.aggregation_)

        return self.classes_.take(np.argmax(y_mu, 1))

    def predict_proba(self, X):

        if not hasattr(self, "classes_"):
            raise Exception("Perform a fit first.")

        X = check_array(X)

        y_mu = predict_protos(X, self.protos_, self.aggregation_)

        return p_normalize(y_mu, 1)  # constrain membership values to probability sum(row) = 1


class OptimizerOWAFactory:

    def __init__(self, optimizer=ga_owa_optimizer()):
        self.optimizer = optimizer
        self.decoder = owa_decoder_plain

    def __call__(self, mus, X, y, class_idx):

        y_target = np.zeros(len(y))
        y_target[y == class_idx] = 1.0

        A = np.zeros(X.shape)  # re-use this matrix

        def fitness(c):
            aggr = self.decoder(c)
            y_pred = predict_proto(X, mus, aggr, A)
            return evaluate_rmse(y_target, y_pred)

        weights = self.optimizer(X, fitness)

        best = self.decoder(weights)

        logger.info("trained owa(%s, %s, %s)" % (str(self.optimizer),
                                                 str(self.decoder).split(" ")[1].split("_")[-1],
                                                 ", ".join(map(lambda x: "%.5f" % (x,), best.v))))

        return best

class static_selection:
    def __init__(self, selection_method):
        self.selection_method = selection_method

    def __call__(self, *args, **kwargs):
        return self.selection_method

class meowa_andness_selection:
    def __init__(self, andness=0.5):
        self.andness = andness

    def __call__(self, X, y):
        return meowa(X.shape[1], self.andness)

class MultipleAggregationsFuzzyPatternClassifier(BaseEstimator, ClassifierMixin):
    """
    Fuzzy pattern classifier with one aggregation for each class.
    """
    def get_params(self, deep=False):
        return {"aggregation_factory": self.aggregation_factory,
                "membership_factory": self.membership_factory,
                "selection_factory": self.selection_factory}

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def __init__(self, membership_factory=pi_factory,
                 aggregation_factory=OptimizerOWAFactory(),
                 selection_factory=static_selection(np.argmax)):
        """
        Constructs object

        Parameters:
        -----------

        membership_factory: the function to use for creating membership function.

        aggregation_factory: the function to use for creating the aggregation function.

        selection_factory: The method of selecting winner in ensemble of classifiers.

        """
        self.aggregation_factory = aggregation_factory
        self.membership_factory = membership_factory
        self.selection_factory = selection_factory

    def fit(self, X, y):

        X = check_array(X)

        self.classes_, y = np.unique(y, return_inverse=True)

        if "?" in tuple(self.classes_):
            raise ValueError("nan not supported for class values")

        # build membership functions for each feature for each class
        learned = [
            learn_class(X, y, y == idx, self.membership_factory, self.aggregation_factory)
            for idx, class_value in enumerate(self.classes_)
        ]

        logger.info("learned %s" % (str(learned),))

        self.protos_ = [ x[0] for x in learned ]
        self.aggregations_ = [ x[1] for x in learned ]
        self.selection_method_ = self.selection_factory(X, y)

        return self

    def predict(self, X):
        """
        Predicts if examples in X belong to classifier's class or not.

        Parameters
        ----------
        X : examples to predict for.
        """
        if not hasattr(self, "classes_"):
            raise Exception("Perform a fit first.")

        y_mu = predict_protos_aggregations(X, self.protos_, self.aggregations_)

        return self.classes_.take(np.argmax(y_mu, 1))

    def predict_proba(self, X):

        if not hasattr(self, "classes_"):
            raise Exception("Perform a fit first.")

        X = check_array(X)

        y_mu = predict_protos_aggregations(X, self.protos_, self.aggregations_)

        return p_normalize(y_mu, 1)  # constrain membership values to probability sum(row) = 1
