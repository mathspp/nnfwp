import numpy as np

def create_weight_matrix(nrows, ncols):
    return np.random.default_rng().normal(loc=0, scale=1/(nrows*ncols), size=(nrows, ncols))

def create_bias_vector(length):
    return create_weight_matrix(length, 1)

def leaky_relu(x, leaky_param = 0.1):
    return np.maximum(x, x*leaky_param)


class Layer:
    def __init__(self, ins, outs, act_function):
        self.ins = ins
        self.outs = outs
        self.act_function = act_function

        self._W = create_weight_matrix(self.outs, self.ins)
        self._b = create_bias_vector(self.outs)

    def forward_pass(self, x):
        return self.act_function(np.dot(self._W, x) + self._b)


if __name__ == "__main__":
    """Demo of chaining layers with compatible shapes."""
    l1 = Layer(2, 4, leaky_relu)
    l2 = Layer(4, 4, leaky_relu)
    l3 = Layer(4, 1, leaky_relu)

    x = np.random.uniform(size=(2, 1))
    output = l3.forward_pass(l2.forward_pass(l1.forward_pass(x)))
    print(output)
