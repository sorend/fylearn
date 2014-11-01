#!/usr/bin/env python

import logging
import paper
import numpy as np
import fylearn.fpcga as fpcga
import fylearn.fuzzylogic as fl
from sklearn import metrics, clone

if __name__ == "__main__":

    logger = logging.getLogger("fpcga")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.FileHandler('info.txt')
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    X, y = paper.load(paper.path(paper.datasets[0]))
    y = y.astype(str)

    l = fpcga.FuzzyPatternClassifierGA(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.prod,), iterations=100, epsilon=None)

    l.fit(X, y)

    y_pred = l.predict(X)
    print metrics.accuracy_score(y_pred, y)

