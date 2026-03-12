import pickle
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    # intinya ini nanti dipake buat menumpuk layer layer yang dibuat
    # TODO : implementasi lengkap

    def __init__(self):
        self.layers = []
        self.history = {'train_loss': [], 'val_loss': []}
        self.loss = None
        self.loss_prime = None

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def save(self, filepath): 
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filepath): 
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
        self.layers = loaded_model.layers
        self.history = loaded_model.history
        self.loss = loaded_model.loss
        self.loss_prime = loaded_model.loss_prime

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        results = []

        for sample in input_data:
            output = sample
            for layer in self.layers:
                output = layer.forward(output)
            results.append(output)

        return np.array(results)

    def train(self, x_train, y_train, epochs, lr):
        if self.loss is None or self.loss_prime is None:
            raise ValueError("Loss function belum diset. GUnakan use(loss, loss_prime) terlebih dahulu.")
        
        samples = len(x_train)
        for epoch in range(epochs):
            err = 0
            for x,y in zip(x_train, y_train):
                output = x
                # forward
                for layer in self.layers:
                    output = layer.forward(output)

                # hitung loss
                err += self.loss(y, output)

                # backward
                error = self. loss_prime(y, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, lr)

            err /= samples
            self.history['train_loss'].append(err)
            print(f"Epoch {epoch+1}/{epochs}, Loss={err}")

    def plot_weight_distribution(self, layer_indices):
        for i in layer_indices:
            if i < 0 or i >= len(self.layers):
                print(f"Layer index {i} tidak valid.")
                continue

            layer = self.layers[i]
            if hasattr(layer, 'weights') and layer.weights is not None:
                plt.figure()
                plt.hist(layer.weights)
                plt.title(f"Weight Distribution - Layer {i}")
                plt.xlabel("Weight value")
                plt.ylabel("Frequency")
                plt.show()
            else:
                print(f"Layer {i} tidak punya weights.")

    def plot_gradient_distribution(self, layer_indices):
        for i in layer_indices:
            if i < 0 or i >= len(self.layers):
                print(f"Layer index {i} tidak valid.")
                continue

            layer = self.layers[i]
            if hasattr(layer, 'weight_gradient') and layer.weights_gradient is not None:
                plt.figure()
                plt.hist(layer.weights_gradient.flatten(), bins=30)
                plt.title(f"Gradient Distribution - Layer {i}")
                plt.xlabel("Gradient value")
                plt.ylabel("Frequency")
                plt.show()
            else:
                print(f"Layer {i} tidak mempunyai weights_gradient.")