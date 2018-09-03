"""
Different cost estimation techniques
------------------------------------
Parameters:
===========
a: actual output
p: predicted output

Return:
=======
cost value of the prediction
"""

import numpy as np


def cross_entropy_cost(a, p):
    m = len(a)
    cost =  (-1 / m) * np.sum(a * np.log(p) + (1 - a) * np.log(1 - p))
    return cost


def linear_cost(a, p):
    delta = a - np.array(p).reshape(len(a), 1)
    return np.mean(delta)


def mean_square(a, p):
    delta = a - np.array(p).reshape(len(a), 1)
    error = np.sum(np.square(delta))
    return error / len(a)