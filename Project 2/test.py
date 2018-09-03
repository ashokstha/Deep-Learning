"""
Code: Main program to execute. Users can also run test.py for the same.
-----------------------------------------------------------------------
Features:
* Customizable type and no. of layers
* Customizable Data sets (MNIST and CIFAR-10)
* Customizable batch size and dropout
* Visualization of Error vs Epochs
"""
import cost_functions as cost
import numpy as np
from ConvNet import ConvNet
from Pool import Pool
from Flatten import Flatten
from FCLayer import FCLayer
from model import Model

def preprocess_data(X):
    return (X-np.mean(X,axis=0))/np.std(X)

def main():
    print("Starting Network...")
    print("-------------------------------------------------------")
    print("Reading Data sets...")
    
    # MNIST Data sets
    #train_img, test_img, train_lbl, test_lbl = load(file_name="mnist")
    
    # CIFAR-10 Data sets
    train_img, test_img, train_lbl, test_lbl = load(file_name="cifar")
    
    Y = train_lbl[:].astype(int)
    X = train_img[:]/255.
    Y_test = test_lbl[:].astype(int)
    X_test = test_img[:]/255.
    
    #preprocess data
    #X = preprocess_data(X)
    #X_test = preprocess_data(X_test)
    
    #model
    model = Model()

    model.add(ConvNet(filter_size=(5,5),filter_no=6,zero_padding=0,stride=(1,1),activation="relu"))
    model.add(Pool(pool_size=(2,2),stride=(2,2),pool_type="max"))
    model.add(ConvNet(filter_size=(5,5),filter_no=6,zero_padding=0,stride=(1,1),activation="relu"))
    model.add(Pool(pool_size=(2,2),stride=(2,2),pool_type="max"))
    model.add(Flatten())
    model.add(FCLayer(activation="relu",n_neurons=32,l_rate=0.001, is_drop_out=True, drop_out=0.7))
    model.add(FCLayer(activation="softmax",n_neurons=10,l_rate=0.001))

    print("-------------------------------------------------------")
    print("CNN Layers:")
    print("-------------------------------------------------------")
    model.print_layers()

    print("-------------------------------------------------------")
    print("Begin Training...")
    
    model.train(X,Y,n_epochs=150, print_loss=True, batch_size=32)
    
    print("End Training.")
    print("-------------------------------------------------------")
    print("Begin Testing...")

    train_accuracy = model.test(X,Y)
    test_accuracy = model.test(X_test,Y_test)

    print("End Testing.")
    print("-------------------------------------------------------")

    print('Training Accuracy: {0:0.2f} %'.format(train_accuracy))
    print('Test Accuracy: {0:0.2f} %'.format(test_accuracy))
    model.show_graph()

if __name__=="__main__":
    main()