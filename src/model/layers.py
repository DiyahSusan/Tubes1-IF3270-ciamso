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
    # ini buat yang fully connected layer
    def __init__(self, input_size, output_size, initializer_func, l1_lambda=0.0, l2_lambda=0.0, **kwargs):
        self.weights = initializer_func((input_size, output_size), **kwargs)
        self.bias = np.zeros((1, output_size)) # biasanya nol di awal, cmiiw

        # regularization parameters
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # dipake buat backward propagation
        self.weights_gradient = None 
        self.bias_gradient = None
        self.input = None

    def forward(self, input_data):
        # ngitung sigma atau Z = XW + b
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        
        # menghitung gradient menggunakan chain rule untuk memperbarui bobot
        input_error = np.dot(output_error, self.weights.T)
        batch_size = self.input.shape[0]

        self.weights_gradient = np.dot(self.input.T, output_error) / batch_size
        self.bias_gradient = np.sum(output_error, axis=0, keepdims=True) / batch_size

        if self.l1_lambda > 0:
            self.weights_gradient += self.l1_lambda * np.sign(self.weights) / batch_size

        if self.l2_lambda > 0:
            self.weights_gradient += self.l2_lambda * self.weights / batch_size

        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient
        
        return input_error
    
    # fungsi tambahan buat ngitung regularization loss
    def get_regularization_loss(self):
        reg_loss = 0.0
        if self.l1_lambda > 0:
            reg_loss += self.l1_lambda * np.sum(np.abs(self.weights))
        if self.l2_lambda > 0:
            reg_loss += self.l2_lambda * np.sum(np.square(self.weights))
        return reg_loss