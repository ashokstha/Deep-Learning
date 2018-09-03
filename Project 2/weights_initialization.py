"""
Different weight initialization techniques
------------------------------------------
Parameters:
===========
p: no. of neurons in previous layer (l-1)
c: no. of neurons in current layer (l)

Return:
=======
weights of the neurons
"""

import numpy as np


def he(p, c):
    return np.random.randn(p, c) * np.sqrt(2.0/p)


def xavier(p, c):
    return np.random.randn(p, c) * np.sqrt(1.0/p)


def _he(p, c):
    return np.random.randn(p, c) * np.sqrt(2.0/ (p + c))
