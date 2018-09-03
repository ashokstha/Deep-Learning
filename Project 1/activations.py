"""
Different Activation functions and their primes
-----------------------------------------------
Parameters:
===========
z: input

Return:
=======
activated value
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    # z = sigmoid(x)
    return z * (1 - z)


def softmax(z):
    z -= np.max(z)
    return (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
    


def softmax_prime(z):
    # z = softmax(x)
    return


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    # z = tanh(x)
    return 1 - z * z


def relu(z):
    return np.maximum(z, 0)


def relu_prime(z):
    dz = np.ones_like(z)
    dz[z < 0] = 0
    return dz


def leaky_relu(z, alpha=0.01):
    return np.maximum(z, z * alpha)


def leaky_relu_prime(z, alpha=0.01):
    dz = np.ones_like(z)
    dz[z < 0] = alpha
    return dz
