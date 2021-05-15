import numpy as np
from abc import ABC, abstractmethod


def create_weight_matrix(nrows, ncols):
    """Create a weight matrix with normally distributed random elements."""
    return np.random.default_rng().normal(loc=0, scale=1/(nrows*ncols), size=(nrows, ncols))

def create_bias_vector(length):
    """Create a bias vector with normally distributed random elements."""
    return create_weight_matrix(length, 1)


class ActivationFunction:
    """Class to be inherited by activation functions."""
    @abstractmethod
    def f(self, x):
        """The method that implements the function."""
        pass

    @abstractmethod
    def df(self, x):
        """Derivative of the function with respect to its input."""
        pass

class LeakyReLU(ActivationFunction):
    """Leaky Rectified Linear Unit."""
    def __init__(self, leaky_param=0.1):
        self.alpha = leaky_param

    def f(self, x):
        return np.maximum(x, x*self.alpha)

    def df(self, x):
        return np.maximum(x > 0, self.alpha)


class LossFunction:
    """Class to be inherited by loss functions."""
    @abstractmethod
    def loss(self, values, expected):
        """Compute the loss of the computed values with respect to the expected ones."""
        pass

    @abstractmethod
    def dloss(self, values, expected):
        """Derivative of the loss with respect to the computed values."""
        pass

class MSELoss(LossFunction):
    """Mean Squared Error Loss function."""
    def loss(self, values, expected):
        return np.mean((values - expected)**2)

    def dloss(self, values, expected):
        return 2*(values - expected)/values.size


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
        return self.act_function.f(np.dot(self._W, x) + self._b)


class NeuralNetwork:
    """A series of connected, compatible layers."""
    def __init__(self, layers, loss):
        self._layers = layers
        self._loss_function = loss

        # Check layer compatibility
        for (from_, to_) in zip(self._layers[:-1], self._layers[1:]):
            if from_.outs != to_.ins:
                raise ValueError("Layers should have compatible shapes.")

    def forward_pass(self, x):
        out = x
        for layer in self._layers:
            out = layer.forward_pass(out)
        return out

    def loss(self, values, expected):
        return self._loss_function.loss(values, expected)


if __name__ == "__main__":
    """Demo of a network as a series of layers."""
    net = NeuralNetwork([
        Layer(2, 4, leaky_relu),
        Layer(4, 4, leaky_relu),
        Layer(4, 1, leaky_relu),
    ])

    x = np.random.uniform(size=(2, 1))
    output = net.forward_pass(x)
    print(output)
