import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np
import cost_functions as cost

class Model(object):
    
    def __init__(self):
        self.layers = []
        self.activations = []      
        self.losses = []
    
    def add(self, layer):
        self.layers.append(layer)
        
    def print_layers(self):
        for layer in self.layers:
            print(layer.__class__.__name__)
    
    def show_graph(self):
        errors = np.array(self.losses)
        plt.plot(errors[:, 0], errors[:, 1], 'r--')
        plt.title("Cost vs Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Training Cost")
        plt.savefig("cost_vs_epochs.png")
        plt.show()
        
    def oneHotIt(self, Y):
        m = Y.shape[0]
        OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
        OHX = np.array(OHX.todense()).T
        return OHX
        
    def forward_pass(self,X,is_test=False):
        activations = []
        activations.append(X)
        
        for layer in self.layers:
            a = layer.forward_pass(activations[-1],is_test)
            activations.append(a)
        
        return activations
    
    def backward_pass(self,X,Y,activations):
        #-------------------- Error ----------------------------
        #calculate loss
        loss = cost.cross_entropy_cost(Y, activations[-1])
        
        #------------------- Backward Pass ----------------------
        m = X.shape[0]
        output = activations[-1]
        delta = Y - output

        layer = self.layers
        for l in range(len(self.layers)-1,-1,-1):
            Act_cLayer = activations[l]
            
            if layer[l].__class__.__name__=="ConvNet":
                delta = layer[l].backward_pass(X=Act_cLayer, delta=delta)
                
            elif layer[l].__class__.__name__=="Pool":
                delta = layer[l].backward_pass(X=Act_cLayer, delta=delta)
                
            elif layer[l].__class__.__name__=="Flatten":
                Act_pLayer = activations[l+1]
                W_pLayer = layer[l+1].weights
                delta= layer[l].backward_pass(delta, Act_pLayer, W_pLayer)
            
            elif layer[l].__class__.__name__=="FCLayer":
                if l==len(self.layers)-1:
                    delta = layer[l].backward_pass(delta, Act_cLayer, is_last_layer=True)
                else:
                    Act_pLayer = activations[1+l]
                    W_pLayer = layer[l+1].weights
                    delta = layer[l].backward_pass(delta, Act_cLayer, Act_pLayer,  W_pLayer) 
            
            else:
                raise Exception("Error! Unknown Layer!")
                
        
        return loss
        
    def train(self, X, Y, n_epochs=100, print_loss=True, batch_size=32):
        y_mat = self.oneHotIt(Y)

        n_batch = int(np.ceil(len(X) / batch_size))
        for i in range(n_epochs):
            #mini batch
            for j in range(n_batch):
                x_mini = X[j * batch_size:(j + 1) * batch_size]
                y_mini = y_mat[j * batch_size:(j + 1) * batch_size]

                activations = self.forward_pass(x_mini)
                loss = self.backward_pass(x_mini,y_mini,activations)
                       
            if print_loss and i%1==0:
                print('Iteration: {0} | Loss: {1}'.format(i,loss))
                self.losses.append([i,loss])

    def getProbsAndPreds(self, X, batch_size=32):
        probs = []
        n_batch = int(np.ceil(len(X) / batch_size))
        for j in range(n_batch):
            x_mini = X[j * batch_size:(j + 1) * batch_size]
            prob = self.forward_pass(x_mini)[-1]
            probs.append(prob)
                         
        probs = np.vstack(probs)
        preds = np.argmax(probs,axis=1) 
        return probs,preds

    def test(self, x,y, batch_size=32):
        prob,prede = self.getProbsAndPreds(x, batch_size)
        accuracy = sum(prede == y)/(float(len(y)))
        return accuracy*100
    