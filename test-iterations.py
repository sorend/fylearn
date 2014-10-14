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

logger = logging.getLogger("fpcga")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.FileHandler('iterations.log')
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
            
L = (
    fpcga.FuzzyPatternClassifierGA(iterations=10, epsilon=None), # all
    fpcga.FuzzyPatternClassifierGA(iterations=50, epsilon=None), # all
    fpcga.FuzzyPatternClassifierGA(iterations=100, epsilon=None), # all
    fpcga.FuzzyPatternClassifierGA(iterations=500, epsilon=None), # all
    fpcga.FuzzyPatternClassifierGA(iterations=1000, epsilon=None), # all
)

# iterate over datsets
import paper
dataset = ("Telugu Vowels", "vowel.csv")
X, y = paper.load(paper.path(dataset))
output = test.execute_one(L, X, y)

print ",".join(dataset) + "," + ",".join(map(str, output))
