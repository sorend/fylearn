#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys
import paper
from scipy.stats import rankdata as rd

data = pd.read_csv(sys.argv[1])

NAMES = map(lambda x: x[0], paper.learners)

print "Classifier", "&", "Mean weighted rank", "&", "$p$-value", r"\\"
print r"\hline"

# iterate datasets
datasets = map(lambda x: x[1], paper.datasets)

# accuracy
A_test  = np.zeros((len(datasets), len(paper.learners)))

# build a_test/s_test/a_train
for d_no, dataset in enumerate(datasets):
    data_idx = data.dataset == dataset
    data_dataset = data[data.dataset == dataset]

    for l_no, learner in enumerate(paper.learners):
        data_d_l = data_dataset[data_dataset.learneridx == l_no]

        A_test[d_no, l_no] = np.mean(data_d_l.test_acc)

print "Data-set," + ",".join(NAMES)
print pd.DataFrame(A_test).to_csv(header=False)
