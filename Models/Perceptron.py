import numpy as np
from ActivationFunction import signum

class Perceptron:
    def __init__(self, learning_rate: float = 1.0, epochs: int = 100, activation=signum, precision: int = 3,bias :float = 0):
        self.weights = None
        self.training_data = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.bias = bias
        self.precision = precision

    def fit(self, training_data, actual_predicted):
        # Set weights to zero like scikit-learn's Perceptron
        number_of_samples, number_of_features = training_data.shape
        self.weights = np.zeros(number_of_features)

        for _ in range(self.epochs):
            for idx, row in enumerate(training_data):
                # Calculate the output with the current weights and bias
                actual_value = actual_predicted[idx]
                linear_output = np.dot(row, self.weights) + self.bias
                predicted_value = self.activation(linear_output)
                prediction_error = actual_value - predicted_value

                # Update weights and bias using the fixed learning rate and no additional scaling
                update_delta = self.learning_rate * prediction_error * row
                self.weights += update_delta

                # Round weights to the specified precision
                self.weights = np.round(self.weights, self.precision)

                # Round bias to the specified precision
                self.bias = round(self.bias + prediction_error * self.learning_rate, self.precision)
        self.weights = np.round(self.weights, 3)
        print(self.weights)

    def predict(self, test):
        linear_output = np.dot(test, self.weights) + self.bias
        predicted_value = self.activation(linear_output)
        return predicted_value
