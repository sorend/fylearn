#!/usr/bin/env python

import logging
import pandas as pd
from joblib import Parallel, delayed
import fylearn.fpcga as fpcga
import fylearn.frr as frr
import numpy as np
import fylearn.fuzzylogic as fl
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import cross_val_score
from sklearn import tree, svm, neighbors
import paper

RUNS = 1

def run_one_classifier(l, X, y):
    test_scores = []
    training_scores = []
    for i in range(RUNS):
        test, training = paper.cross_val_score(l, X, y, cv=10)
        test_scores.extend(test)
        training_scores.extend(training)

    return ( np.mean(test_scores), np.std(test_scores), np.mean(training_scores), np.std(training_scores) )
    
            
def run_one_dataset(logger, L, X, y):
    # cross validation
    scores = Parallel(n_jobs=-1)(delayed(run_one_classifier)(l, X, y) for l in L)
    return scores

if __name__ == "__main__":

    logger = logging.getLogger("fpcga")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.FileHandler('info.txt')
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    L = map(lambda x: x[1], paper.learners)

    def test_acc_std(x):
        return "$%.2f (%.2f)$" % (x[0] * 100.0, x[1] * 100.0)
            
    # iterate over datsets
    import paper
    for dataset in paper.datasets:
        X, y = paper.load(paper.path(dataset))

        output = run_one_dataset(logger, L, X, y)
        print ",".join(dataset) + "," + ",".join(map(test_acc_std, output))
        #print "%s & %s \\\\" % (dataset[0], " & ".join(output))
