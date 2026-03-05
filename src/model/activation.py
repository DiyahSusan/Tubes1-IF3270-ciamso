from src.model.layers import Layer

class ActivationLayer(Layer):
    #  base class untuk aktivasi, tinggal ganti fungsi matematika di dalamnya

    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation            
        self.activation_prime = activation_prime

    def forward(self, input_data):
        pass

    def backward(self, output_error, learning_rate):
        pass
    