import sys, pathlib
# (Ugly) workaround to enable importing from parent folder without too much hassle.
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import csv
import numpy as np
from nn import NeuralNetwork, Layer, LeakyReLU, CrossEntropyLoss, MSELoss

TRAIN_FILE = pathlib.Path(__file__).parent / "mnistdata/mnist_train.csv"
TEST_FILE = pathlib.Path(__file__).parent / "mnistdata/mnist_test.csv"

def load_data(filepath, delimiter=",", dtype=float):
    """Load a numerical numpy array from a file."""

    print(f"Loading {filepath}...")
    with open(filepath, "r") as f:
        data_iterator = csv.reader(f, delimiter=delimiter)
        data_list = list(data_iterator)
    data = np.asarray(data_list, dtype=dtype)
    print("Done.")
    return data

def to_col(x):
    return x.reshape((x.size, 1))

def test(net, test_data):
    correct = 0
    for i, test_row in enumerate(test_data):
        if not i%1000:
            print(i)

        t = test_row[0]
        x = to_col(test_row[1:])/255
        out = net.forward_pass(x)
        guess = np.argmax(out)
        if t == guess:
            correct += 1

    return correct/test_data.shape[0]

def train(net, train_data):
    for i, train_row in enumerate(train_data):
        if not i%1000:
            print(i)

        net.train(to_col(train_row[1:])/255, train_row[0])

def train_students(teacher, students, train_data):
    for i, train_row in enumerate(train_data):
        if not i%1000:
            print(i)

        x = to_col(train_row[1:])/255
        out = teacher.forward_pass(x)
        for student in students:
            student.train(x, out)


if __name__ == "__main__":
    layers = [
        Layer(784, 16, LeakyReLU()),
        Layer(16, 16, LeakyReLU()),
        Layer(16, 10, LeakyReLU()),
    ]
    teacher = NeuralNetwork(layers, CrossEntropyLoss(), 0.03)
    students = [
        NeuralNetwork([Layer(784, 10, LeakyReLU())], MSELoss(), 0.001),
        NeuralNetwork([Layer(784, 10, LeakyReLU())], MSELoss(), 0.003),
        NeuralNetwork([Layer(784, 10, LeakyReLU())], MSELoss(), 0.01),
        NeuralNetwork([Layer(784, 10, LeakyReLU())], MSELoss(), 0.03),
        NeuralNetwork([Layer(784, 10, LeakyReLU())], MSELoss(), 0.1),
        NeuralNetwork([Layer(784, 10, LeakyReLU())], MSELoss(), 0.3),
    ]

    test_data = load_data(TEST_FILE, delimiter=",", dtype=int)
    accuracy = test(teacher, test_data)
    print(f"Accuracy is {100*accuracy:.2f}%")     # Expected to be around 10%

    train_data = load_data(TRAIN_FILE, delimiter=",", dtype=int)
    train(teacher, train_data)

    accuracy = test(teacher, test_data)
    print(f"Accuracy is {100*accuracy:.2f}%")

    print("Training students.")
    train_students(teacher, students, train_data)
    print("Testing students.")
    accuracies = [100*test(student, test_data) for student in students]
    print(accuracies)
    print(f"Teacher accuracy had been {100*accuracy:.2f}%")
