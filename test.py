#!/usr/bin/env python

import logging
import pandas as pd
import fylearn.fpcga as fpcga
import fylearn.frr as frr
import numpy as np
import fylearn.fuzzylogic as fl
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import cross_val_score
from sklearn import tree, svm, neighbors

RUNS = 1
            
def execute_one(logger, L, X, y):

    # iterate learners
    output = []
    for l in L:
        # cross validation
        test_scores = []
        training_scores = []
        for i in range(RUNS):
            test, training = paper.cross_val_score(l, X, y, cv=10)
            test_scores.extend(test)
            training_scores.extend(training)

        output.extend([ np.mean(test_scores), np.std(test_scores), np.mean(training_scores), np.std(training_scores) ])
        #print "---"
        #print "dataset", dataset
        logger.info("learner=%s test=%s mean=%f std=%f, training=%s mean=%f std=%f" % (str(l), str(test_scores), np.mean(test_scores), np.std(test_scores), str(training_scores), np.mean(training_scores), np.std(training_scores)))

    return output

if __name__ == "__main__":

    logger = logging.getLogger("fpcga")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.FileHandler('info.txt')
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    L = (
        tree.DecisionTreeClassifier(),
        svm.SVC(kernel='linear'),
        neighbors.KNeighborsClassifier(),
        frr.FuzzyReductionRuleClassifier(aggregation=fl.prod),
        frr.FuzzyReductionRuleClassifier(aggregation=fl.mean),
        fpcga.FuzzyPatternClassifierGA(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.prod,), iterations=50),
        fpcga.FuzzyPatternClassifierGA(mu_factories=(fpcga.build_pi_membership,), aggregation_rules=(fl.mean,), iterations=50),
        fpcga.FuzzyPatternClassifierGA(iterations=50), # all
    )
            
    # iterate over datsets
    import paper
    for dataset in paper.datasets:
        X, y = paper.load(paper.path(dataset))

        output = execute_one(L, X, y)
        print ",".join(dataset) + "," + ",".join(map(str, output))
        #print "%s & %s \\\\" % (dataset[0], " & ".join(output))
