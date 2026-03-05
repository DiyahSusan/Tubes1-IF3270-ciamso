import numpy as np

class Layer:
    # intinya ini biar tiap layernya bisa melakukan forward dan backward
    # ini base class, nanti implementasiin sesuai kebutuhan

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError
    

class DenseLayer(Layer):
    def __init__(self, input_size, output_size, initializer="random_uniform", seed=None):
        self.weights = None 
        self.bias = None
        self.weights_gradient = None 
        self.bias_gradient = None
        pass

    def forward(self, input_data):
        pass

    def backward(self, output_error, learning_rate):
        pass