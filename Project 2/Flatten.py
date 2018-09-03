import numpy as np

class Flatten(object):
    
    def __init__(self):
        pass
    
    def forward_pass(self, X, is_test=False):
        self.size = X.shape
        batch, z, x, y = X.shape
        a = np.reshape(X,(batch, x*y*z))
        return a
    
    def backward_pass(self, delta, Act_pLayer, W_pLayer):
        delta = np.resize(delta,self.size)
        return delta