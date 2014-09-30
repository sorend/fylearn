#!/usr/bin/env python

import numpy as np
import paper

for dataset in paper.datasets:
    X, y = paper.load(paper.path(dataset))

    print dataset[0], "&", X.shape[0], "&", X.shape[1], "&", len(np.unique(y)), "& \\\\"

