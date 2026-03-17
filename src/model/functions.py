# ini isinya math things nya, semua rumus ada disini
import numpy as np

# TODO : implementasikan sesuai kebutuhan

# fungsi aktivasi dan turunan pertamanya
class ActivationFunctions:
    def __init__(self):
        pass

    def linear(self, x):
        return x

    def linear_prime(self, x):
        return np.ones_like(x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1/(1+ np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def hyperbolic_tangent(self, x):
        return np.tanh(x)

    def hyperbolic_tangent_prime(self, x):
        return 1 - np.tanh(x)**2

    def softmax(self, x):
        x = np.array(x)
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def softmax_prime(self, x):
        s = self.softmax(x)
        batch_jacobian = []

        for row in s:
            row = row.reshape(-1, 1)
            jacobian = np.diagflat(row) - np.dot(row, row.T)
            batch_jacobian.append(jacobian)

        return np.array(batch_jacobian)

# loss function
class LossFunctions:
    def __init__(self):        
        pass    
    
    def mse(self, y_true, y_pred): 
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_prime(self, y_true, y_pred): 
        return 2 * (y_pred - y_true)

    def binary_cross_entropy(self, y_true, y_pred): 
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

    def bce_prime(self, y_true, y_pred): 
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        n = y_true.size
        
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))

    def categorical_cross_entropy(self, y_true, y_pred): 
        n = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / n
                
    def cce_prime(self, y_true, y_pred): 
        n = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -(y_true / y_pred)