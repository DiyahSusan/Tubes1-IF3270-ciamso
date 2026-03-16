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

    def get_total_regularization_loss(self):
        reg_loss = 0.0
        for layer in self.layers:
            if hasattr(layer, 'get_regularization_loss'):
                reg_loss += layer.get_regularization_loss()
        return reg_loss

    def predict(self, input_data):
        output = input_data.copy()
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, x_val=None, y_val=None, 
            epochs=10, batch_size=32, learning_rate=0.01, verbose=1):
        
        self.history = {'train_loss': [], 'val_loss': []}

        n_samples = len(x_train)
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            train_loss = 0.0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]

                # forward
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)

                # data loss
                batch_data_loss = self.loss(y_batch, output)

                # backward + update
                error = self.loss_prime(y_batch, output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

                batch_reg_loss = self.get_total_regularization_loss()
                batch_total_loss = batch_data_loss + batch_reg_loss

                train_loss += batch_total_loss * len(x_batch)

            train_loss /= n_samples
            self.history['train_loss'].append(train_loss)

            val_loss = None
            if x_val is not None and y_val is not None:
                val_output = x_val.copy()
                for layer in self.layers:
                    val_output = layer.forward(val_output)

                val_data_loss = self.loss(y_val, val_output)
                val_reg_loss = self.get_total_regularization_loss()
                val_loss = val_data_loss + val_reg_loss

                self.history['val_loss'].append(val_loss)

            if verbose == 1:
                msg = f"Epoch {epoch+1}/{epochs}, Train Loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f", Val Loss={val_loss:.6f}"
                print(msg)

        return self.history

    def plot_weight_distribution(self, layer_indices):
        for i in layer_indices:
            if i < 0 or i >= len(self.layers):
                print(f"Layer index {i} tidak valid.")
                continue

            layer = self.layers[i]
            if hasattr(layer, 'weights') and layer.weights is not None:
                plt.figure(figsize=(6, 4))
                plt.hist(layer.weights.flatten(), bins=30)
                plt.title(f"Weight Distribution - Layer {i}")
                plt.xlabel("Weight value")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.show()
            else:
                print(f"Layer {i} tidak punya weights.")

    def plot_gradient_distribution(self, layer_indices):
        for i in layer_indices:
            if i < 0 or i >= len(self.layers):
                print(f"Layer index {i} tidak valid.")
                continue

            layer = self.layers[i]
            if hasattr(layer, 'weights_gradient') and layer.weights_gradient is not None:
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.hist(layer.weights_gradient.flatten(), bins=30)
                plt.title(f"Weight Gradient - Layer {i}")
                plt.xlabel("Gradient value")
                plt.ylabel("Frequency")

                # plot bias gradient
                plt.subplot(1, 2, 2)
                if hasattr(layer, 'bias_gradient') and layer.bias_gradient is not None:
                    plt.hist(layer.bias_gradient.flatten(), bins=30)
                    plt.title(f"Bias Gradient - Layer {i}")
                    plt.xlabel("Gradient value")
                    plt.ylabel("Frequency")
                else:
                    plt.text(0.5, 0.5, "No bias gradient", ha='center', va='center')
                    plt.title(f"Bias Gradient - Layer {i}")
                
                plt.tight_layout()
                plt.show()
            else:
                print(f"Layer {i} tidak mempunyai weights_gradient.")