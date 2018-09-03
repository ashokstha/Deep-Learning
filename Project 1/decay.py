import numpy as np


def step_decay(epoch, initial_lrate=0.1, drop=0.6, epochs_drop=1000):
    return initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))


def exp_decay(epoch, initial_lrate=0.1, k=0.1):
    return initial_lrate * np.exp(-k * epoch)
