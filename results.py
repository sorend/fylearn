#!/usr/bin/env python

import numpy as np
import pandas as pd
import sys
import paper

def test_accuracy(data, idx):
    return int(data[idx] * 10000.0)

def prepare_accuracy(data, idx, best_acc):
    dacc = int(data[idx] * 10000.0)
    test_acc = data[idx] * 100.0
    test_std = data[idx+1] * 100.0
    train_acc = data[idx+2]
    train_std = data[idx+3]
    fmt = "$%.2f(%.2f)$" if dacc != best_acc else "$\mathbf{%.2f(%.2f)}$"
    return fmt % (test_acc, test_std)

data = pd.read_csv(sys.argv[1], header=None)

NAMES = map(lambda x: x[0], paper.learners)

print "\\#", "&", " & ".join(NAMES), "\\\\"
print "\\hline"

# each row
for i in range(len(data)):
    row = data.ix[i,:]
    idxs = range(2, len(row), 4)

    best_acc = max([ test_accuracy(row, idx) for idx in idxs ])
    
    vals = []
    for j in range(len(idxs)):
        vals.append(prepare_accuracy(row, idxs[j], best_acc))

    print (i+1), "&", " & ".join(vals), " \\\\"
    
