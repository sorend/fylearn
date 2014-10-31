#!/usr/bin/env python

import logging
import pandas as pd
import fylearn.fpcga as fpcga
import fylearn.frr as frr
import numpy as np
import fylearn.fuzzylogic as fl
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import cross_val_score
import test
import sklearn.tree as tree

if __name__ == "__main__":
    logger = logging.getLogger("fpcga")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.FileHandler('iterations.log')
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
            
    L = (
        ("FPC_10", fpcga.FuzzyPatternClassifierGA(iterations=10, epsilon=None, mu_factories=(fpcga.build_pi_membership,))),
        ("FPC_50", fpcga.FuzzyPatternClassifierGA(iterations=50, epsilon=None, mu_factories=(fpcga.build_pi_membership,))),
        ("FPC_100", fpcga.FuzzyPatternClassifierGA(iterations=100, epsilon=None, mu_factories=(fpcga.build_pi_membership,))),
        ("FPC_500", fpcga.FuzzyPatternClassifierGA(iterations=500, epsilon=None, mu_factories=(fpcga.build_pi_membership,))),
        ("FPC_1000", fpcga.FuzzyPatternClassifierGA(iterations=1000, epsilon=None, mu_factories=(fpcga.build_pi_membership,))),
        ("SFPC_10", fpcga.FuzzyPatternClassifierGA2(iterations=10, epsilon=None, mu_factories=(fpcga.build_pi_membership,))),
        ("SFPC_50", fpcga.FuzzyPatternClassifierGA2(iterations=50, epsilon=None, mu_factories=(fpcga.build_pi_membership,))),
        ("SFPC_100", fpcga.FuzzyPatternClassifierGA2(iterations=100, epsilon=None, mu_factories=(fpcga.build_pi_membership,))),
        ("SFPC_500", fpcga.FuzzyPatternClassifierGA2(iterations=500, epsilon=None, mu_factories=(fpcga.build_pi_membership,))),
        ("SFPC_1000", fpcga.FuzzyPatternClassifierGA2(iterations=1000, epsilon=None, mu_factories=(fpcga.build_pi_membership,))),
    )

    # iterate over datsets
    import paper
    for dataset in paper.datasets:
        X, y = paper.load(paper.path(dataset))
        test.run_one_dataset(dataset[0], L, X, y)
