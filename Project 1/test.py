#main program to execute

import matplotlib.pyplot as plt
from ann_classifier import Network as Network

def main():
    print("Starting Network...")
    print("-------------------------------------------------------")
    print("Reading Data sets...")
    
    # MNIST Data sets
    train_img, test_img, train_lbl, test_lbl = load(file_name="mnist")
    
    # CIFAR-10 Data sets
    #train_img, test_img, train_lbl, test_lbl = load(file_name="cifar")
    
    y = train_lbl[:].astype(int)
    x = train_img[:]/255.
    testY = test_lbl[:].astype(int)
    testX = test_img[:]/255.
    
    n_in = x.shape[1]
    n_out = 10
    
    print("-------------------------------------------------------")
    print("Training started...")
    
    nn = Network(n_in=n_in, n_out=n_out, l_rate=0.9)
    nn.add_layer(activation_function="tanh", n_neurons=25)
    nn.add_layer(activation_function="tanh", n_neurons=17)
    nn.add_layer(activation_function="softmax", n_neurons=n_out)
    nn.train(x,y, n_epoch=50, print_loss=True)
    
    print("Training ended.")
    print("-------------------------------------------------------")
    print("Testing started...")

    train_accuracy = nn.getAccuracy(x,y)
    test_accuracy = nn.getAccuracy(testX,testY)

    print("Testing ended.")
    print("-------------------------------------------------------")
    
    print('Training Accuracy: {0:0.2f} %'.format(train_accuracy))
    print('Test Accuracy: {0:0.2f} %'.format(test_accuracy))
    
    #nn.show_graph()
    
if __name__ =="__main__":
    main()

