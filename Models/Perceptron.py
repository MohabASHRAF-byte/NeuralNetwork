import numpy as np
from ActivationFunction import signum
from utilities import confusion_matrix


class Perceptron:
    def __init__(self, learning_rate: float = 1.0, epochs: int = 100, activation=signum, precision: int = 3,
                 bias: bool = False):
        self.weights = None
        self.training_data = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.UseBias = bias
        self.bias = 0
        self.precision = precision

    def fit(self, training_data, actual_predicted):
        # Set weights to zero
        number_of_samples, number_of_features = training_data.shape
        self.weights = np.zeros(number_of_features)

        for _ in range(self.epochs):
            for idx, row in enumerate(training_data):
                # Calculate the output with the current weights and bias
                actual_value = actual_predicted[idx]
                predicted_value = self.net_input(row)
                prediction_error = actual_value - predicted_value

                # Update weights and bias using the fixed learning rate and no additional scaling
                update_delta = np.round(self.learning_rate * prediction_error * row, self.precision)
                self.weights += update_delta

                # Round weights to the specified precision
                self.weights = np.round(self.weights, self.precision)
                # Round bias to the specified precision
                if self.UseBias:
                    self.bias = round(self.bias + prediction_error * self.learning_rate, self.precision)


        return self.weights,self.bias
    def net_input(self, data_input):
        linear_output = np.dot(data_input, self.weights) + self.bias
        predicted_value = self.activation(linear_output)
        return predicted_value

    def predict(self, test):
        linear_output = np.dot(test, self.weights) + self.bias
        predicted_value = self.activation(linear_output)
        return predicted_value

    def confusion_matrix(self, test_input, test_output):
        classified = self.predict(test_input)
        # this is implemented confusion matrix in utilities not built in
        return confusion_matrix(classified,test_output)

    def run(self):
        pass