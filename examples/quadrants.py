import sys, pathlib
# (Ugly) workaround to enable importing from parent folder without too much hassle.
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from nn import NeuralNetwork, Layer, LeakyReLU, MSELoss
# import matplotlib.pyplot as plt
import numpy as np

def col(a):
    """Make sure the array is a column."""
    return np.atleast_2d(a).T

N = 100_000     # Create N points inside the square [-2,2]×[-2,2]
data = np.random.uniform(low=-2, high=2, size=(2, N))

ts = np.zeros(shape=(2, N))
ts[0, data[0,]*data[1,]>0] = 1
ts[1, :] = 1 - ts[0, :]

net = NeuralNetwork([
    Layer(2, 3, LeakyReLU()),
    Layer(3, 2, LeakyReLU()),
], MSELoss(), 0.05)

def assess(net, data, ts):
    correct = 0
    # cs = []
    for i in range(data.shape[1]):
        out = net.forward_pass(col(data[:, i]))
        guess = np.argmax(np.ndarray.flatten(out))
        if ts[guess, i]:
            correct += 1
        # cs.append(guess)
    # fig = plt.figure()
    # plt.scatter(data[0, :1000], data[1, :1000], c=cs)
    # fig.show()
    # input()
    return correct

test_to = 1000
print(assess(net, data[:, :test_to], ts[:, :test_to]))
for i in range(test_to, N):
    net.train(col(data[:, i]), col(ts[:, i]))
print(assess(net, data[:, :test_to], ts[:, :test_to]))
