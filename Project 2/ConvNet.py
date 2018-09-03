import numpy as np
from Layers import Layers

class ConvNet(object):
    def __init__(self, filter_size=(3, 3), filter_no=6, zero_padding=0, stride=(1, 1),
                 activation="relu", l_rate=0.1):
        self.filter_size = filter_size
        self.filter_no = filter_no
        self.zero_padding = zero_padding
        self.stride = stride
        self.activation = activation
        self.weights = []
        self.biases = 0
        self.l_rate = l_rate

        self.layer = Layers(activation_function=activation)
        x, y = filter_size
        self.init_weights(filter_no, x, y)

    def init_weights(self, z, x, y):
        # init filter/weights and bias
        self.weights = np.random.rand(z, x, y)
        self.biases = 0

    def padding(self, X, p):
        #padding in x,y || X: batch * channel * x * y
        pad_width = ((0, 0),(0, 0),(p, p), (p, p))
        return np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)
        
    def forward_pass(self, X, is_test=False):
        fx, fy = self.filter_size
        sx, sy = self.stride
        p = self.zero_padding
        W = self.weights
        B = self.biases
        
        #adding padding to input data
        if p>0:
            X = self.padding(X,p)
            
        batch, len_z, len_x, len_y = X.shape

        # output size
        len_zx = (len_x + 2 * p - fx) // sx + 1
        len_zy = (len_x + 2 * p - fy) // sy + 1
        len_zz = self.filter_no
        Z = np.zeros([batch, len_zz, len_zx, len_zy])
        
        for k in range(self.filter_no):
            for i in range(0, len_x - fx, sx):
                for j in range(0, len_y - fy, sy):
                    Z[:,k, i // sx, j // sy] = np.sum(X[:, :, i:i + fx, j:j + fy] * W[k] + B)

        # activation function after ConvNet
        A = self.layer.activation_function(Z)
        return A

    def backward_pass(self, X, delta):
        fx, fy = self.filter_size
        sx, sy = self.stride
        p = self.zero_padding
        W = self.weights
        B = self.biases

        batch, len_z, len_x, len_y = delta.shape
        # output size
        z, x, y = W.shape
        dW = np.zeros([z, x, y])
        delta = self.layer.activation_prime_function(delta)
        dX = np.zeros(X.shape)
        
        for b in range(batch):
            for k in range(len_z):
                for i in range(0, len_x, sx):
                    for j in range(0, len_y, sy):
                        dX[b,:, i:i + fx, j:j + fy] += W[k] * delta[b,k,i,j]
                        dW[k] += X[b,0, i:i + fx, j:j + fy] * delta[b,k,i,j]
                    
        # update parameters
        dW = np.reshape(dW,(z, x, y))
        self.weights = W - dW
        
        return dX
