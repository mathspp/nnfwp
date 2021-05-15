import numpy as np

def create_weight_matrix(nrows, ncols):
    return np.random.default_rng().normal(loc=0, scale=1/(nrows*ncols), size=(nrows, ncols))

def create_bias_vector(length):
    return create_weight_matrix(length, 1)
