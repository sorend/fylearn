#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# load data
data = pd.read_csv("iterations.txt", header=None)

# take accuracy test results
acc = data.ix[:,2::4]
std = data.ix[:,3::4]

iters = (10, 50, 100, 500, 1000)

lines = ("r-", "g-", "b-", "c-", "m-", "y-", "k-", "r--", "g--", "b--", "c--", "m--")

for i in range(12):
    plt.plot(iters, acc.ix[i], lines[i], label=str(i+1))

from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
plt.legend(prop = fontP, loc=4)

plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.ylim((0.4, 1.0))

plt.savefig("iterations-graph.pdf")
