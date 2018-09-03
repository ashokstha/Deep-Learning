from Layers import Layers
import decay
import numpy as np

class FCLayer(object):
    
    def __init__(self, activation="relu", n_neurons=120, l_rate=0.1, lrate_decay="step_decay", is_drop_out=False, 
                 drop_out=0.5, momentum=0.8):
        self.weights = []
        self.biases = 0
        self.n_epochs = 0
        self.n_neurons = n_neurons
        self.initial_lrate = l_rate
        self.l_rate = l_rate
        self.is_drop_out = is_drop_out
        self.drop_out = []
        self.dropout_p = drop_out
        self.momentum_coefficient = momentum
        self.v = 0
        self.m = 0
        
        self.layer = Layers(activation_function=activation)
        self.set_lrate_decay(lrate_decay=lrate_decay)
        
    def init_weights(self, p, c):
        self.weights = self.layer.weight_function(p, c)
        self.biases = np.zeros((1, c))
        
    def set_lrate_decay(self, lrate_decay):
        if lrate_decay == "step_decay":
            self.lrate_decay = decay.step_decay
        elif lrate_decay == "exp_decay":
            self.lrate_decay = decay.exp_decay
            
    def forward_pass(self, X, is_test=False):
        #initialize weights for the first time if lenght is 0
        if len(self.weights) == 0:
            in_x = X.shape[1]
            self.init_weights(p=in_x,c=self.n_neurons)
        
        W = self.weights
        B = self.biases
        epsilon = 1e-6
        Z = X.dot(W) + B + epsilon
        
        #activation function after ConvNet
        A = self.layer.activation_function(Z)
        
        if self.is_drop_out==True and is_test==False:
                p = self.dropout_p
                dropout = np.random.binomial(1, p, size=Z.shape)
                self.dropout = dropout
                A *= dropout
        return A
        
    def backward_pass(self, delta, Act_cLayer, Act_pLayer=None, W_pLayer=None, l_rate=0.1, is_last_layer=False):
        n = delta.shape[0]
        
        self.n_epochs += 1
        if (self.n_epochs+1)%100 == 0:
                self.l_rate = self.lrate_decay(self.initial_lrate)
                
        if is_last_layer == False:
            prime = self.layer.activation_prime_function(Act_pLayer)
            delta = np.dot(delta, W_pLayer.T) * prime
              
        if self.is_drop_out==True:
                delta *= self.dropout
                
        #update parameters
        dw = (1/n) * np.dot(Act_cLayer.T,delta)
        
        #Adam optimization
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = self.n_epochs
        m = self.m
        v = self.v
        
        m = beta1*m + (1-beta1)*dw
        mt = m / (1-beta1**t)
        self.m = m
        
        v = beta2*v + (1-beta2)*(dw**2)
        vt = v / (1-beta2**t)
        self.v = v
        
        self.weights += self.l_rate * mt / (np.sqrt(vt) + eps)
        
        db = (1/n) * np.sum(delta, axis=0, keepdims=True)
        self.biases += self.l_rate * db
        return delta
    