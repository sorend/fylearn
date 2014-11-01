#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys
from scipy.stats import rankdata as rd
import paper

NAMES = map(lambda x: x[0], paper.learners)

data = pd.read_csv(sys.argv[1], header=None)

ranks = np.zeros((len(data), len(NAMES)))
X = np.array(data)
X = X[:,2::4]
X = 1.0 - X

# each row
for ri in range(len(X)):
    X[ri,:] = rd(X[ri,:])

y = np.mean(X, 0)

print "Classifier", "&", "Average rank", "\\\\"
print "\\hline"
for i in range(len(NAMES)):
    print NAMES[i], "&", "%.2f" % (y[i],), "\\\\"
