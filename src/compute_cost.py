import numpy as np


def cost_function(X: np.array, Y: np.array, theta: np.array, alpha):
    m = len(Y)
    h = X.dot(theta)
    return ((h - Y) ** 2).sum() / (2 * m)
