# ini isinya math things nya, semua rumus ada disini
import numpy as np

# TODO : implementasikan sesuai kebutuhan

# fungsi aktivasi dan turunan pertamanya
def linear(x):
    return x

def linear_prime(x):
    return 1

def relu(x):
    return np.max(0, x)

def relu_prime(x):
    if x > 0:
        return 1
    else:
        return 0

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def hyperbolic_tangent(x):
    return np.tanh(x)

def hyperbolic_tangent_prime(x):
    return (2 / (np.exp(x) - np.exp(-x)))**2

def softmax(x):
    x = np.array(x)
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x/np.sum(exp_x)

def softmax_prime(x):
    s = softmax(x).reshape(-1,1)
    return np.diagflat(s) - np.dot(s,s.T)

# loss function
def mse(y_true, y_pred): 
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred): 
    return 2 * (y_pred - y_true) / y_true.size

def binary_cross_entropy(y_true, y_pred): 
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

def bce_prime(y_true, y_pred): 
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return ((1-y_true)/(1-y_pred) - (y_true/y_pred)/y_true.size)

def categorical_cross_entropy(y_true, y_pred): 
    n = y_true.shape[0]
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / n
            
def cce_prime(y_true, y_pred): 
    n = y_true.shape[0]
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -(y_true / y_pred)/n