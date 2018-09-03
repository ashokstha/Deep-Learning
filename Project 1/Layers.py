import numpy as np
import activations
import weights_initialization


class Layers(object):

    def __init__(self, n_in, n_out=4, activation_function="tanh"):
        self.n_in = n_in
        self.n_out = n_out
        self.set_activation_functions(act_function_name=activation_function)

    def set_activation_functions(self, act_function_name="tanh"):
        if act_function_name == "relu":
            self.activation_function = activations.relu
            self.activation_prime_function = activations.relu_prime
            self.set_weight_function(weight_name="he")
        elif act_function_name == "leaky_relu":
            self.activation_function = activations.leaky_relu
            self.activation_prime_function = activations.leaky_relu_prime
            self.set_weight_function(weight_name="he")
        elif act_function_name == "sigmoid":
            self.activation_function = activations.sigmoid
            self.activation_prime_function = activations.sigmoid_prime
            self.set_weight_function(weight_name="xavier")
        elif act_function_name == "softmax":
            self.activation_function = activations.softmax
            self.activation_prime_function = activations.softmax_prime
            self.set_weight_function(weight_name="he")
        elif act_function_name == "tanh":
            self.activation_function = activations.tanh
            self.activation_prime_function = activations.tanh_prime
            self.set_weight_function(weight_name="xavier")
        else:
            raise Exception("Error! Activation function not found!")

    def set_weight_function(self, weight_name):
        if weight_name == "he":
            self.weight_function = weights_initialization.he
        elif weight_name == "xavier":
            self.weight_function = weights_initialization.xavier
        elif weight_name == "_he":
            self.weight_function = weights_initialization._he
        else:
            raise Exception("Error! Weight function not found!")
