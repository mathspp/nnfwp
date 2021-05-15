import numpy as np

def create_weight_matrix(nrows, ncols):
    """Create a weight matrix with normally distributed random elements."""
    return np.random.default_rng().normal(loc=0, scale=1/(nrows*ncols), size=(nrows, ncols))

def create_bias_vector(length):
    """Create a bias vector with normally distributed random elements."""
    return create_weight_matrix(length, 1)

def leaky_relu(x, leaky_param = 0.1):
    """Leaky Rectified Linear Unit with default parameter."""
    return np.maximum(x, x*leaky_param)


class Layer:
    """Model the connections between two sets of neurons in a network."""
    def __init__(self, ins, outs, act_function):
        self.ins = ins
        self.outs = outs
        self.act_function = act_function

        self._W = create_weight_matrix(self.outs, self.ins)
        self._b = create_bias_vector(self.outs)

    def forward_pass(self, x):
        """Compute the next set of neuron states with the given set of states."""
        return self.act_function(np.dot(self._W, x) + self._b)


class NeuralNetwork:
    """A series of connected, compatible layers."""
    def __init__(self, layers):
        self._layers = layers

        # Check layer compatibility
        for (from_, to_) in zip(self._layers[:-1], self._layers[1:]):
            if from_.outs != to_.ins:
                raise ValueError("Layers should have compatible shapes.")

    def forward_pass(self, x):
        out = x
        for layer in self._layers:
            out = layer.forward_pass(out)
        return out


if __name__ == "__main__":
    """Demo of chaining layers with compatible shapes."""
    l1 = Layer(2, 4, leaky_relu)
    l2 = Layer(4, 4, leaky_relu)
    l3 = Layer(4, 1, leaky_relu)

    x = np.random.uniform(size=(2, 1))
    output = l3.forward_pass(l2.forward_pass(l1.forward_pass(x)))
    print(output)
