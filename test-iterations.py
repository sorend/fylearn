#!/usr/bin/env python

import logging
import pandas as pd
import fylearn.fpcga as fpcga
import fylearn.frr as frr
import numpy as np
import fylearn.fuzzylogic as fl
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import cross_val_score
from test import execute_one
import sklearn.tree as tree

logger = logging.getLogger("fpcga")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.FileHandler('iterations.log')
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
            
L = (
    fpcga.FuzzyPatternClassifierGA(iterations=10, epsilon=0.00000001), # all
    fpcga.FuzzyPatternClassifierGA(iterations=50, epsilon=0.00000001), # all
    fpcga.FuzzyPatternClassifierGA(iterations=100, epsilon=0.00000001), # all
    fpcga.FuzzyPatternClassifierGA(iterations=500, epsilon=0.00000001), # all
    fpcga.FuzzyPatternClassifierGA(iterations=1000, epsilon=0.00000001), # all
)

# iterate over datsets
import paper
dataset = ("Telugu Vowels", "vowel.csv")
X, y = paper.load(paper.path(dataset))
output = execute_one(L, X, y)

print ",".join(dataset) + "," + ",".join(map(str, output))
