import numpy as np

# initializers sesuai spek

class Initializers:
    def __init__(self):
        pass

    def zero_init(self, shape): 
        # isi dengan nol
        return np.zeros(shape)

    def uniform_init(self, shape, low, high, seed=None): 
        # random dengan distribusi uniform
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low, high, size=shape)

    def normal_init(self, shape, mean, variance, seed=None): 
        # random dengan distribusi normal
        if seed is not None:
            np.random.seed(seed)
        
        # standar deviasi
        std_dev = np.sqrt(variance)
        return np.random.normal(mean, std_dev, size=shape)

    # bonus kalo mau
    def xavier_init(self, shape, seed=None): 
        if seed is not None:
            np.random.seed(seed)
        
        n_in, n_out = shape
        limit = np.sqrt(6.0 / (n_in + n_out))
        return np.random.uniform(-limit, limit, size=shape)

    def he_init(self, shape, seed=None): 
        if seed is not None:
            np.random.seed(seed)
        
        n_in, n_out = shape
        std = np.sqrt(2.0 / n_in)
        return np.random.normal(0, std, size=shape)