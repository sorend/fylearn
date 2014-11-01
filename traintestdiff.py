#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys
import paper

data = pd.read_csv(sys.argv[1])

# iterate datasets
datasets = map(lambda x: x[1], paper.datasets)

# accuracy and std
A_test  = np.zeros((len(datasets), len(paper.learners)))

# accuracy training data
A_train = np.zeros(A_test.shape)

# build a_test/s_test/a_train
for d_no, dataset in enumerate(datasets):
    data_dataset = data[data.dataset == dataset]

    for l_no, learner in enumerate(paper.learners):
        data_d_l = data_dataset[data_dataset.learneridx == l_no]

        A_test[d_no, l_no] = np.mean(data_d_l.test_acc)
        A_train[d_no, l_no] = np.mean(data_d_l.training_acc)

A_diff = A_train - A_test

NAMES = map(lambda x: x[0], paper.learners)

print "\\#", "&", " & ".join(NAMES), "\\\\"
print "\\hline"

for ridx, row in enumerate(A_diff):
    print ridx + 1, "&", " & ".join(map(lambda x: "$%.2f$" % (x * 100.0,), row)), r"\\"

print r"\hline"
print "Avg. &", " & ".join(map(lambda x: "$%.2f$" % (x,), np.mean(A_diff * 100.0, 0))), r"\\"
