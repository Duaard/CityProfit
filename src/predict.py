import numpy as np


def predict(population, theta):
    # Normalize given population
    population /= 10000
    x = np.array([1, population])
    return x.transpose().dot(theta)
