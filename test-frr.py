#!/usr/bin/env python

import pandas as pd
import fylearn.frr as frr
import numpy as np
import fylearn.fuzzylogic as fl
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("./wine.csv")
X = data.ix[:,:8]
y = data["class"]

# scale to [0, 1]
X = MinMaxScaler().fit_transform(X)

l = frr.FuzzyReductionRuleClassifier(aggregation=fl.mean)

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(l, X, y, cv=10)

print "scores %s, mean %f, std %f" % \
 (str(scores), np.mean(scores), np.std(scores))
