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

RUNS  = 1
FOLDS = 10

def run_one_classifier(key, l, X, y):
    test_scores = []
    training_scores = []
    for i in range(RUNS):
        # scores is a tuple of (test, training) accuracy
        scores = paper.cross_val_score(l, X, y, cv=FOLDS, n_jobs=-1)
        for j in range(FOLDS):
            print "%s,%d,%d,%f,%f" % (key, i, j, scores[j][0], scores[j][1])

    #return ( np.mean(test_scores), np.std(test_scores), np.mean(training_scores), np.std(training_scores) )
            
def run_one_dataset(key, L, X, y):
    for lidx, l in enumerate(L):
        run_one_classifier("\"%s\",\"%s\",%d" % (key, l[0], lidx), l[1], X, y)

if __name__ == "__main__":

    logger = logging.getLogger("fpcga")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.FileHandler('info.txt')
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #def test_acc_std(x):
    #    return "$%.2f (%.2f)$" % (x[0] * 100.0, x[1] * 100.0)

    # iterate over datsets
    print "dataset,learner,learneridx,iteration,fold,test_acc,training_acc"
    
    import paper
    for dataset in paper.datasets:
        X, y = paper.load(paper.path(dataset))

        run_one_dataset(dataset[1], paper.learners, X, y)
        # print ",".join(dataset) + "," + ",".join([ str(i) for j in output for i in j ])
        #print "%s & %s \\\\" % (dataset[0], " & ".join(output))
