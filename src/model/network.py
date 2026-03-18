import pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from model.layers import DenseLayer
from model.activation import ActivationLayer
from model.functions import ActivationFunctions
from model.initializers import Initializers

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

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=32, learning_rate=0.01, verbose=1, optimizer="gd", beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.history = {'train_loss': [], 'val_loss': []}

        n_samples = len(x_train)
        n_batches = int(np.ceil(n_samples / batch_size))
        t = 0 # timestamp adam

        epoch_logs = []
        
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
                t += 1
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate, optimizer=optimizer, t=t, beta1=beta1, beta2=beta2, epsilon=epsilon)

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
            
            msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f}"
            if val_loss is not None:
                msg += f" | Val Loss: {val_loss:.6f}"
            epoch_logs.append(msg)

            # progress bar
            if verbose == 1:
                progress = (epoch + 1) / epochs
                bar_length = 20
                filled = int(bar_length * progress)
                bar = "#" * filled + "-" * (bar_length - filled)

                clear_output(wait=True)
                print(f"[{bar}] {epoch+1}/{epochs}")
                print(msg)

        if verbose == 1:
            clear_output(wait=True)
            print("Training selesai.\n")
            for log in epoch_logs:
                print(log)

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


class FFNN(NeuralNetwork):
    def __init__(self, layers, activations, init_func='xavier', l1=0.0, l2=0.0, seed=None, seed_mode="same", mean=0.0, variance=1.0, low=-0.05, high=0.05):
        super().__init__()

        if not isinstance(layers, (list, tuple)) or len(layers) < 2:
            raise ValueError("parameter 'layers' harus berupa list/tuple dengan minimal 2 elemen, contoh: [64, 16, 1].")
        if len(activations) != len(layers) - 1:
            raise ValueError("panjang list 'activations' harus sama dengan jumlah layer transformasi (panjang layers - 1).")

        self._activation_functions = ActivationFunctions()
        self._initializers = Initializers()
        self.mean = mean
        self.variance = variance
        self.low = low
        self.high = high
        initializer_list = self._check_list_init_func(init_func, len(layers) - 1)

        for i in range(len(layers) - 1):
            input_size = layers[i]
            output_size = layers[i + 1]

            init_func = self._check_init_func(initializer_list[i])
            layer_seed = None
            if seed is not None:
                if seed_mode == "same":
                    layer_seed = seed
                elif seed_mode == "incremental":
                    layer_seed = seed + i
                else:
                    raise ValueError("seed_mode harus 'same' atau 'incremental'")

            self.add(DenseLayer(input_size, output_size, init_func, l1_lambda=l1, l2_lambda=l2, seed=layer_seed))
            activation_func, activation_prime_func = self._check_activation_func(activations[i])
            self.add(ActivationLayer(activation_func, activation_prime_func))

    def _check_list_init_func(self, init_func, n_layers):
        if isinstance(init_func, (list, tuple)):
            if len(init_func) != n_layers:
                raise ValueError("'init_func' harus memiliki panjang yang sama dengan jumlah layer transformasi.")
            return list(init_func)
        return [init_func] * n_layers

    def _check_init_func(self, initializer):
        if isinstance(initializer, str):
            if initializer == 'he':
                return lambda shape, seed=None: self._initializers.he_init(shape, seed=seed)
            if initializer == 'xavier':
                return lambda shape, seed=None: self._initializers.xavier_init(shape, seed=seed)
            if initializer == 'normal':
                return lambda shape, seed=None: self._initializers.normal_init(shape, mean=self.mean, variance=self.variance, seed=seed)
            if initializer == 'uniform':
                return lambda shape, seed=None: self._initializers.uniform_init(shape, low=self.low, high=self.high, seed=seed)
            if initializer == 'zero':
                return lambda shape, seed=None: self._initializers.zero_init(shape)
        raise ValueError(f"Initializer `{initializer}` tidak ditemukan.")

    def _check_activation_func(self, activation_func):
        if isinstance(activation_func, str):
            if activation_func == 'tanh':
                activation_func = 'hyperbolic_tangent'

            if activation_func in ['linear', 'relu', 'sigmoid', 'hyperbolic_tangent', 'softmax', 'leaky_relu', 'elu', 'swish']:
                prime = f"{activation_func}_prime"    
                if hasattr(self._activation_functions, activation_func) and hasattr(self._activation_functions, prime):
                    return getattr(self._activation_functions, activation_func), getattr(self._activation_functions, prime)
        raise ValueError(f"Activation function `{activation_func}` tidak ditemukan.")