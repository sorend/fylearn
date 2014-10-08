#!/usr/bin/env python

import logging
import pandas as pd
import fylearn.fpcga as fpcga
import fylearn.frr as frr
import numpy as np
import fylearn.fuzzylogic as fl
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import cross_val_score
import sklearn.tree as tree

logger = logging.getLogger("fpcga")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.FileHandler('info.txt')
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
            
def execute_one(L, X, y):

    # iterate learners
    output = []
    for l in L:
        # cross validation
        scores = []
        for i in range(10):
            one = cross_val_score(l, X, y, cv=10)
            scores.extend(one)

        output.append("$%.2f \pm %.2f$" % (np.mean(scores), np.std(scores)))
        #print "---"
        #print "dataset", dataset
        logger.info("learner %s scores %s, mean %f, std %f" % (str(l), str(scores), np.mean(scores), np.std(scores)))

    return output


L = (
    tree.DecisionTreeClassifier(),
    frr.FuzzyReductionRuleClassifier(aggregation=fl.mean),
    frr.FuzzyReductionRuleClassifier(aggregation=fl.prod),
    fpcga.FuzzyPatternClassifierGA(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.prod,), iterations=50),
    fpcga.FuzzyPatternClassifierGA(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.mean,), iterations=50),
    fpcga.FuzzyPatternClassifierGA(iterations=50), # all
)
            
# iterate over datsets
import paper
for dataset in paper.datasets:
    X, y = paper.load(paper.path(dataset))
    output = execute_one(L, X, y)
    print "%s & %s \\\\" % (dataset[0], " & ".join(output))
