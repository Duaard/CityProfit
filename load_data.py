import numpy as np


def load_file(filename):
    f = open(filename, "r")
    X = []
    Y = []
    for line in f:
        [x, y] = line.split(',')
        X.append(float(x))
        Y.append(float(y))
    return [X, Y]
