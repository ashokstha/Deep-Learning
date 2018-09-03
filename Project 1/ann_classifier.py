"""
MLP implementaion
-----------------------------------------------
Parameters:
===========
n_in: no. of input units
n_out: no. of output units
l_rate: learning rate
cost_function: name of cost function
lrate_decay: name of learning rate decay


Return:
=======
object
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.sparse

import cost_functions
import activations
import decay
from Layers import Layers

class Network(object):
    
    def __init__(self, n_in=784, n_out=10, l_rate=0.1,
                 cost_function="cross-entropy",
                 lrate_decay="step_decay"):

        self.n_in = n_in
        self.n_out = n_out
        self.initial_lrate = l_rate
        self.l_rate = l_rate
        self.momentum_coefficient = 0.5

        self.weights = []
        self.biases = []
        self.previous_weights = []
        self.previous_biases = []
        self.layers = []
        self.losses = []
        self.v = []
        self.dropout_p = []
        self.dropout = []
        self.is_dropout = []
        
        self.set_cost_function(cost_function_name=cost_function)
        self.set_lrate_decay(lrate_decay=lrate_decay)

    def add_layer(self, activation_function="tanh", n_neurons=4, is_dropout=False,drop_out=1):
        if len(self.layers) <= 0:
            n_previous_neurons = self.n_in
        else:
            n_previous_neurons = self.layers[-1].n_out
            
        self.is_dropout.append(is_dropout)
        self.dropout_p.append(drop_out)

        layer = Layers(n_in=n_previous_neurons, n_out=n_neurons, activation_function=activation_function)
        self.layers.append(layer)
        
    def oneHotIt(self, Y):
        m = Y.shape[0]
        OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
        OHX = np.array(OHX.todense()).T
        return OHX
    
    def show_graph(self):
        errors = np.array(self.losses)
        plt.plot(errors[:, 0], errors[:, 1], 'r--')
        plt.title("Cost vs Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Training Cost")
        plt.savefig("cost_vs_epochs.png")
        plt.show()
        
    def set_lrate_decay(self, lrate_decay):
        if lrate_decay == "step_decay":
            self.lrate_decay = decay.step_decay
        elif lrate_decay == "exp_decay":
            self.lrate_decay = decay.exp_decay

    def set_cost_function(self, cost_function_name):
        if (cost_function_name == "cross-entropy"):
            self.cost_function = cost_functions.cross_entropy_cost
        elif (cost_function_name == "linear"):
            self.cost_function = cost_functions.linear_cost
        elif (cost_function_name == "mean-square"):
            self.cost_function = cost_functions.mean_square
        else:
            raise Exception("Error! Cost function not found!")
    
    def getProbsAndPreds(self, x):
        probs = self.forward_propagation(x)[-1]
        preds = np.argmax(probs,axis=1)      
        return probs,preds

    def getAccuracy(self, x,y, is_test=True):
        if is_test:
            for l in range(len(self.layers)):
                self.is_dropout[l] = False
        
        prob,prede = self.getProbsAndPreds(x)
        accuracy = sum(prede == y)/(float(len(y)))
        return accuracy*100
    
    def forward_propagation(self, X):
        a = [X]
        for l in range(len(self.layers)):
            z = a[l].dot(self.weights[l]) + self.biases[l]
            activation = self.layers[l].activation_function(z)

            if self.is_dropout[l]==True:
                p = self.dropout_p[l]
                dropout = np.random.binomial(1, p, size=z.shape)
                self.dropout[l] = dropout
                activation *= dropout

            a.append(activation)

        return a
    
    def back_propagation(self, x,y_mat,a):
        m = x.shape[0]
        output = a[-1]

        #loss = (-1 / m) * np.sum(y_mat * np.log(output))
        loss = self.cost_function(y_mat, output)
        
        deltas = []
        delta = y_mat - output
        
        #for last layer
        dw = (1/m) * np.dot(a[-2].T,delta)
        beta = self.momentum_coefficient
        self.v[-1] = beta * self.v[-1] + (1-beta)*dw
        self.weights[-1] += self.l_rate * self.v[-1]
            
        db = (1/m) * np.sum(delta, axis=0, keepdims=True)
        self.biases[-1] += self.l_rate * db
        
        #for remaining layers
        for l in range(len(self.layers)-2, -1, -1):
            prime = self.layers[l].activation_prime_function(a[l+1])
            w = self.weights[l+1]
            delta = np.dot(delta, w.T) * prime
            
            if self.is_dropout[l]==True:
                delta *= self.dropout[l]
            
            dw = (1/m) * np.dot(a[l].T,delta)
            beta = self.momentum_coefficient
            self.v[l] = beta * self.v[l] + (1-beta)*dw
            self.weights[l] += self.l_rate * self.v[l]
            
            db = (1/m) * np.sum(delta, axis=0, keepdims=True)
            self.biases[l] += self.l_rate * db
        
        return loss
    
    
    def init_weights(self):
        for i in range(len(self.layers)):
            n_cur_layer_neurons = self.layers[i].n_out
            n_prev_layer_neurons = self.layers[i].n_in

            weights = self.layers[i].weight_function(n_prev_layer_neurons, n_cur_layer_neurons)
            self.weights.append(weights)

            biases = np.zeros((1, n_cur_layer_neurons))
            self.biases.append(biases)
            self.v.append(0)
            
            self.dropout.append(1)
                

    def train(self, x, y, n_epoch=100, print_loss=True, batch_size=512):
        
        self.init_weights()
        
        y_mat = self.oneHotIt(y)
        
        n_batch = int(np.ceil(len(x) / batch_size))
        for i in range(n_epoch):
            #mini batch
            for j in range(n_batch):
                x_mini = x[j * batch_size:(j + 1) * batch_size]
                y_mini = y_mat[j * batch_size:(j + 1) * batch_size]
                
                a = self.forward_propagation(x_mini)
                loss = self.back_propagation(x_mini, y_mini,a)
                       
            if print_loss and i%10==0:
                print('Iteration: {0} | Loss: {1}'.format(i,loss))
                self.losses.append([i,loss])
            
            if (i+1)%100 == 0:
                self.l_rate = self.lrate_decay(self.initial_lrate)