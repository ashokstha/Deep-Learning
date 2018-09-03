"""
Different weight initialization techniques
------------------------------------------
Parameters:
===========
p: no. of neurons in previous layer (l+1)
c: no. of neurons in current layer (l)

Return:
=======
weights of the neurons
"""

import numpy as np


def he(p, c):
    np.random.seed(0)
    return np.random.rand(p, c) * np.sqrt(2/p)


def xavier(p, c):
    return np.random.rand(p, c) / np.sqrt(p)


def _he(p, c):
    np.random.seed(0)
    return np.random.rand(p, c) * np.sqrt(2/ (p + c))

