# Testing
from Data import Data
import numpy as np
import matplotlib.pyplot as plt
from ActivationFunction import unit_step, signum
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def test_function_Data():
    ob = Data()
    train_input, train_output, test_input, test_output = ob.GenerateData(['A', 'B'])
    print(train_input)
    print(train_output)
    print(test_input)
    print(test_output)


def test_perceptron_1():
    from Models.Perceptron import Perceptron

    X, y = datasets.make_blobs(
        n_samples=int(1e3), n_features=5, centers=2, cluster_std=1.05, random_state=2
    )
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, epochs=1000, activation=unit_step)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()


def test_implemented_perceptron(train_input, train_output, test_input, test_output):
    from Models.Perceptron import Perceptron
    print(type(train_input))
    model = Perceptron(learning_rate=0.001, activation=signum, epochs=100, bias=7)
    model.fit(train_input, train_output)
    predictions = model.predict(test_input)

    print("Perceptron classification accuracy", accuracy(test_output, predictions))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # plt.scatter(train_input[:, 0], train_input[:, 1], marker="o", c= test_output)
    #
    # x0_1 = np.amin(X_train[:, 0])
    # x0_2 = np.amax(X_train[:, 0])
    #
    # x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    # x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
    #
    # ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
    #
    # ymin = np.amin(X_train[:, 1])
    # ymax = np.amax(X_train[:, 1])
    # ax.set_ylim([ymin - 3, ymax + 3])
    #
    # plt.show()


def test_builtin_perceptron(train_input, train_output, test_input, test_output):
    from sklearn.linear_model import Perceptron

    # Display the type of train_input to confirm it's a numpy array
    print("Type of train_input:", type(train_input))

    # Initialize the built-in Perceptron model
    model = Perceptron(eta0=0.001, max_iter=1000)

    # Train the model on the training data
    model.fit(train_input, train_output)

    # Predict the test data
    predictions = model.predict(test_input)

    # Calculate and display accuracy
    accuracy_score_result = accuracy_score(test_output, predictions)
    print("Perceptron classification accuracy:", accuracy_score_result)

    # Display weights and bias for inspection
    print("Model Weights:", model.coef_)
    print("Model Bias:", model.intercept_)


def oursVsBuiltin_perceptron():
    data = Data()
    train_input, train_output, test_input, test_output = data.GenerateData(['B', 'C'])
    test_implemented_perceptron(train_input, train_output, test_input, test_output)
    print('#' * 100)
    test_builtin_perceptron(train_input, train_output, test_input, test_output)

def test_implemented_Adaline(train_input, train_output, test_input, test_output):
    from Models.Adaline import ADALINE
    print(type(train_input))
    model = ADALINE(learning_rate=0.001,  epochs=100)
    model.fit(train_input, train_output)
    predictions = model.predict(test_input)

    print("Perceptron classification accuracy", accuracy(test_output, predictions))

def oursVsBuiltin_Adaline():
    data = Data()
    train_input, train_output, test_input, test_output = data.GenerateData(['B', 'C'])
    test_implemented_Adaline(train_input, train_output, test_input, test_output)

