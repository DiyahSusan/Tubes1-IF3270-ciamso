import numpy as np

# initializers sesuai spek

def zero_init(shape): 
    # isi dengan nol
    return np.zeros(shape)

def uniform_init(shape, low, high, seed=None): 
    # random dengan distribusi uniform
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, size=shape)

def normal_init(shape, mean, variance, seed=None): 
    # random dengan distribusi normal
    if seed is not None:
        np.random.seed(seed)
    
    # standar deviasi
    std_dev = np.sqrt(variance)
    return np.random.normal(mean, std_dev, size=shape)

# bonus kalo mau
def xavier_init(shape): 
    pass
def he_init(shape): 
    pass