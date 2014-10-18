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

    # iterate learners
    output = []
    for l in L:
        # cross validation
        scores = Parallel(n_jobs=-1)(delayed(paper.cross_val_score)(l, X, y, cv=10) for l in L)
        output.extend(scores)

        #print "---"
        #print "dataset", dataset
        logger.info("learner=%s mean=%f std=%f, mean=%f std=%f" % (str(l), scores[0], scores[1], scores[2], scores[3]))

    return output

if __name__ == "__main__":

    logger = logging.getLogger("fpcga")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.FileHandler('info.txt')
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    L = map(lambda x: x[1], paper.learners)
            
    # iterate over datsets
    import paper
    for dataset in paper.datasets:
        X, y = paper.load(paper.path(dataset))

        output = run_one_dataset(logger, L, X, y)
        print ",".join(dataset) + "," + ",".join(map(str, output))
        #print "%s & %s \\\\" % (dataset[0], " & ".join(output))
