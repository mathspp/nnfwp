import numpy as np

def create_weight_matrix(nrows, ncols):
    return np.random.default_rng().normal(loc=0, scale=1/(nrows*ncols), size=(nrows, ncols))

def create_bias_vector(length):
    return create_weight_matrix(length, 1)

def leaky_relu(x, leaky_param = 0.1):
    return np.maximum(x, x*leaky_param)
