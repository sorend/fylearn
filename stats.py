#!/usr/bin/env python

import numpy as np
import paper

for i, dataset in enumerate(paper.datasets):
    X, y = paper.load(paper.path(dataset))

    u, c = np.unique(y, return_inverse=True)
    dummy = float(np.max(np.bincount(c))) / len(y)
    print (i+1), "&", dataset[0], "&", X.shape[0], "&", X.shape[1], "&", len(np.unique(y)), "&", "%.2f" % (dummy,), "&  \\\\"

