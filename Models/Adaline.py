import numpy as np
from prettytable import PrettyTable

from ActivationFunction import signum
from utilities import confusion_matrix


class AdalineGD:
    def __init__(self, learning_rate=0.001, epochs=50, threshold=0.01, activation=signum, bias=False):
        self.weights = None
        self.training_input_std = None
        self.training_input_mean = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.activation = activation
        self.bias = 0
        self.UseBias = bias

    def fit(self, training_input, training_output):

        # Standardize the input features and save the mean and std for later use To prevent overflow
        self.training_input_mean = training_input.mean(axis=0)
        self.training_input_std = training_input.std(axis=0)

        # Execute standardization
        training_input = (training_input - self.training_input_mean) / self.training_input_std

        # initialize the weights with zeros
        number_of_samples, number_of_features = training_input.shape
        self.weights = np.zeros(number_of_features)

        for epoch in range(self.epochs):
            cost_epoch = 0
            for training_row, target in zip(training_input, training_output):
                # Get predication for the current weights
                output = self.net_input(training_row)
                # Calc the error
                error = target - output
                # Update the weights && bias based on the previous error
                self.weights += self.learning_rate * error * training_row
                if self.UseBias:
                    self.bias += self.learning_rate * error
                cost_epoch += (error ** 2) * .5

            # Calculate MSE and return if it under the passed threshold
            mse = cost_epoch / number_of_samples
            if mse < self.threshold:
                break
        return self.weights, self.bias

    def net_input(self, training_input):
        return np.dot(training_input, self.weights) + self.bias

    def predict(self, training_input):
        # Standardize training_input using the training mean and std
        training_input = (training_input - self.training_input_mean) / self.training_input_std
        # Calculate the output of the model 
        linear_output = self.net_input(training_input)
        # Get the output of the activation function 
        classified_classes = self.activation(linear_output)
        return classified_classes

    def confusion_matrix(self, data_input, data_output):
        # Get predicted classes
        classified = self.predict(data_input)
        # this is implemented confusion matrix in utilities not built in
        return confusion_matrix(data_output, classified)
