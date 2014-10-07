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

def execute_one(L, X, y):

    # iterate learners
    output = []
    for l in L:
        # cross validation
        scores = cross_val_score(l, X, y, cv=10)

        output.append("$%.2f \pm %.2f$" % (np.mean(scores), np.std(scores)))
        print "---"
        print "dataset", dataset
        print "learner", l

        print "scores %s, mean %f, std %f" % \
            (str(scores), np.mean(scores), np.std(scores))

    return output


logger = logging.getLogger("fpcga")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.FileHandler('info.txt')
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
            
L = (
    fpcga.FuzzyPatternClassifierGA(iterations=10), # all
    fpcga.FuzzyPatternClassifierGA(iterations=50), # all
    fpcga.FuzzyPatternClassifierGA(iterations=100), # all
    fpcga.FuzzyPatternClassifierGA(iterations=500), # all
    fpcga.FuzzyPatternClassifierGA(iterations=1000), # all
)
            
# iterate over datsets
import paper
dataset = ("Telugu Vowels", "vowel.csv")
X, y = paper.load(paper.path(dataset))
output = execute_one(L, X, y)
print "%s & %s \\\\" % (dataset[0], " & ".join(output))
