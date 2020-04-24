#!/usr/bin/env python3

''' Usage:

$ python3 knn.py <ord> [<k1> <k2> ... ]

<ord> is the order of the norm: 1 Manhattan; 2 Euclidean (actually can be anything)
<k1> <k2> ... is a list of k's that you wanna try (as many as you want).

e.g.,

$ python3 knn.py 1 1 3 5

dimensionality: 92
k=1    train_err  1.5% (+: 25.1%) dev_err 23.50% (+: 26.9%)
k=3    train_err 11.5% (+: 24.0%) dev_err 19.40% (+: 25.4%)
k=5    train_err 13.9% (+: 24.0%) dev_err 17.50% (+: 24.7%)

$ python3 knn.py 2 1 3 5

dimensionality: 92
k=1    train_err  1.5% (+: 25.1%) dev_err 23.30% (+: 27.1%)
k=3    train_err 11.5% (+: 24.0%) dev_err 19.20% (+: 25.8%)
k=5    train_err 13.7% (+: 24.3%) dev_err 17.80% (+: 25.0%)
'''

import sys
from collections import defaultdict
import numpy as np

def process_data(filename):
    X, Y = [], []
    for j, line in enumerate(open(filename)):
        line = line.strip()
        features = line.split(", ")
        feat_vec = np.zeros(dimension)
        for i, fv in enumerate(features[:-1]): # last one is target
            if i in [0,7]: # two numerical fields
                feat_vec[feature_map[i, 0]] = float(fv) / 50  # NB: diff 2 not 1!
            elif (i, fv) in feature_map: # ignore unobserved features
                feat_vec[feature_map[i, fv]] = 1

        X.append(feat_vec)
        Y.append(1 if features[-1] == ">50K" else -1) # fake for testdata

    return np.array(X), np.array(Y)

def knn(k, example, train):
    trainX, trainY = train
    # neighbors = np.argsort(np.linalg.norm(example - trainX, axis=1), k)[:k]
    neighbors = np.argpartition(np.linalg.norm(example - trainX, axis=1, ord=ord), k)[:k]
    votes = trainY[neighbors] # slicing
    return 1 if sum(votes) > 0 else -1

def eval(k, testX, testY, train):
    pred = np.array([knn(k, vecx, train) for vecx in testX])
    errors = sum(pred != testY)
    positives = sum(pred == 1)
    return errors / len(testX) * 100, positives / len(testX) * 100

if __name__ == "__main__":
    field_value_freqs = defaultdict(lambda : defaultdict(int)) # field_id -> value -> freq
    for line in open("income.train.txt.5k"):
        line = line.strip()
        features = line.split(", ")[:-1] # exclude target label
        for i, fv in enumerate(features):
            field_value_freqs[i][0 if i in [0,7] else fv] += 1

    feature_map = {}
    feature_remap = {}
    for i, value_freqs in field_value_freqs.items():
        for v in value_freqs:
            k = len(feature_map) # bias
            feature_map[i, v] = k
            feature_remap[k] = i, v

    dimension = len(feature_map) # bias
    print("dimensionality: %d" % dimension) #, feature_map

    train_data = process_data("income.train.txt.5k") 
    dev_data = process_data("income.dev.txt")
    # test_data = process_data("income.test.txt")

    ord = int(sys.argv[1]) # order: 1 (Manhattan) or 2 (Euclidean) or any p
    for k in map(int, sys.argv[2:]):
        train_err, train_pos = eval(k, train_data[0][:5000], train_data[1][:5000], train_data)
        dev_err, dev_pos = eval(k, dev_data[0][:1000], dev_data[1][:1000], train_data)

        print ("k=%-4d train_err %4.1f%% (+:%5.1f%%) dev_err %4.2f%% (+:%5.1f%%)" % (k, train_err, train_pos, dev_err, dev_pos))
