import numpy as np

class Pool(object):

    def __init__(self, pool_size=(3, 3), zero_padding=0, stride=(1, 1), pool_type="max"):
        self.pool_size = pool_size
        self.zero_padding = 0 if zero_padding < 0 else zero_padding
        self.stride = stride

        self.set_pooling(pool_type)

    def set_pooling(self, pool_type="max"):
        if pool_type == "max":
            self.pool_type = np.max
        elif pool_type == "average":
            self.pool_type = np.average
        else:
            raise Exception("Error! Pooling type not found!")
            
    def padding(self, X, p):
        #padding in x,y || X: batch * channel * x * y
        pad_width = ((0, 0),(0, 0),(p, p), (p, p))
        return np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)

    def forward_pass(self, X, is_test=False):

        fx, fy = self.pool_size
        sx, sy = self.stride
        p = self.zero_padding
        
        #adding padding to input data
        if p>0:
            X = self.padding(X,p)

        batch, len_z, len_x, len_y = X.shape  # len_z: channel

        # output size
        len_zx = (len_x + 2 * p - fx) // sx + 1
        len_zy = (len_x + 2 * p - fy) // sy + 1
        len_zz = len_z
        Z = np.zeros([batch, len_zz, len_zx, len_zy])

        for k in range(len_z):
            for i in range(0, len_x - fx, sx):
                for j in range(0, len_y - fy, sy):
                    Z[:, k, i // sx, j // sy] = self.pool_type(X[:, k, i:i + fx, j:j + fy])

        return Z

    def nanargmax(self, a):
        idx = np.argmax(a, axis=None)
        multi_idx = np.unravel_index(idx, a.shape)
        
        if np.isnan(a[multi_idx]):
            multi_idx = (0,0)
        return multi_idx

    def backward_pass(self, X, delta):

        fx, fy = self.pool_size
        sx, sy = self.stride
        p = self.zero_padding
        batch, len_z, len_x, len_y = X.shape

        # output size
        dZ = np.zeros([batch, len_z, len_x, len_y])

        for b in range(batch):
            for k in range(len_z):
                for i in range(0, len_x - fx, sx):
                    for j in range(0, len_y - fy, sy):
                        x, y = self.nanargmax(X[b,k, i:i + fx, j:j + fy])
                        dZ[b,k, i + x, i + y] = delta[b,k, i // sx, j // sy]

        return dZ
