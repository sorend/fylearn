#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys
import paper
from scipy.stats import rankdata as rd

data = pd.read_csv(sys.argv[1])

NAMES = map(lambda x: x[0], paper.learners)

print "\\#", "&", " & ".join(NAMES), "\\\\"
print "\\hline"

# iterate datasets
datasets = map(lambda x: x[1], paper.datasets)

# accuracy and std
A_test  = np.zeros((len(datasets), len(paper.learners)))
S_test  = np.zeros(A_test.shape)

# accuracy training data
A_train = np.zeros(A_test.shape)

# build a_test/s_test/a_train
for d_no, dataset in enumerate(datasets):
    data_idx = data.dataset == dataset
    data_dataset = data[data.dataset == dataset]

    for l_no, learner in enumerate(paper.learners):
        data_d_l = data_dataset[data_dataset.learneridx == l_no]

        A_test[d_no, l_no] = np.mean(data_d_l.test_acc)
        S_test[d_no, l_no] = np.std(data_d_l.test_acc)
        A_train[d_no, l_no] = np.mean(data_d_l.training_acc)

for ridx, row in enumerate(A_test):
    O = []
    ranked = rd(row)
    best_i = np.argwhere(ranked == np.amax(ranked)).flatten().tolist()

    for cidx, col in enumerate(row):
        if cidx in best_i:
            O.append(r"$\mathbf{%.2f (%.2f)}$" % (col * 100.0, S_test[ridx, cidx] * 100.0))
        else:
            O.append(r"$%.2f (%.2f)$" % (col * 100.0, S_test[ridx, cidx] * 100.0))

    print "%d" % (ridx + 1), "&", " & ".join(O), r"\\"

