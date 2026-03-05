import pickle
import numpy as np

class NeuralNetwork:
    # intinya ini nanti dipake buat menumpuk layer layer yang dibuat
    # TODO : implementasi lengkap

    def __init__(self):
        self.layers = []
        self.history = {'train_loss': [], 'val_loss': []}

    def save(self, filepath): 
        pass

    def load(self, filepath): 
        pass

    def add(self, layer):
        pass

    def predict(self, input_data):
        pass

    def train(self, x_train, y_train, epochs, lr):
        pass

    def plot_weight_distribution(self, layer_indices):
        pass

    def plot_gradient_distribution(self, layer_indices):
        pass