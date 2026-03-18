from model.layers import Layer

class ActivationLayer(Layer):
    #  base class untuk aktivasi, tinggal ganti fungsi matematika di dalamnya
    # ini dipake buat yang non linear

    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation            
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data # simpan untuk backward pass 
        self.output = self.activation(self.input) # terapkan fungsi aktivasi 
        return self.output

    def backward(self, output_error, learning_rate, **kwargs):
        return self.activation_prime(self.input) * output_error