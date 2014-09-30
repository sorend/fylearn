#!/usr/bin/env python

import pandas as pd
import fylearn.fpcga as fpcga
import fylearn.frr as frr
import numpy as np
import fylearn.fuzzylogic as fl
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import cross_val_score
import sklearn.tree as tree

def execute_one(L, X, y):

    # iterate learners
    for l in L:
        # cross validation    
        scores = cross_val_score(l, X, y, cv=10)

        print "---"
        print "dataset", dataset
        print "learner", l

        print "scores %s, mean %f, std %f" % \
            (str(scores), np.mean(scores), np.std(scores))


L = (
    tree.DecisionTreeClassifier(),
    frr.FuzzyReductionRuleClassifier(aggregation=np.nanmean),
    frr.FuzzyReductionRuleClassifier(aggregation=fl.prod),
    fpcga.FuzzyPatternClassifierGA(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.prod,)),
    fpcga.FuzzyPatternClassifierGA(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.mean,)),
    fpcga.FuzzyPatternClassifierGA(), # all
)
            
# iterate over datsets
import paper
for dataset in paper.datasets:
    X, y = paper.load(paper.path(dataset))
    execute_one(L, X, y)
