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

        # state utk adam
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)

    def forward(self, input_data):
        # ngitung sigma atau Z = XW + b
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_error, learning_rate, optimizer="gd", t=1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        # menghitung gradient menggunakan chain rule untuk memperbarui bobot
        input_error = np.dot(output_error, self.weights.T)
        batch_size = self.input.shape[0]

        self.weights_gradient = np.dot(self.input.T, output_error) / batch_size
        self.bias_gradient = np.sum(output_error, axis=0, keepdims=True) / batch_size

        if self.l1_lambda > 0:
            self.weights_gradient += self.l1_lambda * np.sign(self.weights) / batch_size

        if self.l2_lambda > 0:
            self.weights_gradient += self.l2_lambda * self.weights / batch_size

        if optimizer == "gd":
            self.weights -= learning_rate * self.weights_gradient
            self.bias -= learning_rate * self.bias_gradient

        elif optimizer == "adam":
            # momentum pertama
            self.m_w = beta1 * self.m_w + (1 - beta1) * self.weights_gradient
            self.m_b = beta1 * self.m_b + (1 - beta1) * self.bias_gradient

            # momentum kedua
            self.v_w = beta2 * self.v_w + (1 - beta2) * (self.weights_gradient ** 2)
            self.v_b = beta2 * self.v_b + (1 - beta2) * (self.bias_gradient ** 2)

            # bias correction
            m_w_hat = self.m_w / (1 - beta1 ** t)
            m_b_hat = self.m_b / (1 - beta1 ** t)
            v_w_hat = self.v_w / (1 - beta2 ** t)
            v_b_hat = self.v_b / (1 - beta2 ** t)

            self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        else:
            raise ValueError(f"Optimizer '{optimizer}' tidak dikenali.")

        return input_error
    
    # buat ngitung regularization loss
    def get_regularization_loss(self):
        reg_loss = 0.0
        if self.l1_lambda > 0:
            reg_loss += self.l1_lambda * np.sum(np.abs(self.weights))
        if self.l2_lambda > 0:
            reg_loss += 0.5 * self.l2_lambda * np.sum(np.square(self.weights))
        return reg_loss

class RMSNormLayer(Layer):
    def __init__(self, feature_size, eps=1e-8):
        super().__init__()
        self.feature_size = feature_size
        self.eps = eps

        # parameter trainable
        self.gamma = np.ones((1, feature_size))
        self.gamma_gradient = None

        # cache forward
        self.input = None
        self.rms = None
        self.normalized = None

    def forward(self, input_data):
        self.input = input_data

        mean_square = np.mean(np.square(input_data), axis=1, keepdims=True)
        self.rms = np.sqrt(mean_square + self.eps)
        self.normalized = input_data / self.rms
        self.output = self.gamma * self.normalized

        return self.output

    def backward(self, output_error, learning_rate, optimizer="gd", t=1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        batch_size = self.input.shape[0]
        d = self.input.shape[1]

        self.gamma_gradient = np.sum(output_error * self.normalized, axis=0, keepdims=True) / batch_size
        dxhat = output_error * self.gamma
        dot = np.sum(dxhat * self.input, axis=1, keepdims=True)
        input_error = (dxhat / self.rms) - (self.input * dot) / (d * (self.rms ** 3))

        # update gamma
        self.gamma -= learning_rate * self.gamma_gradient

        return input_error