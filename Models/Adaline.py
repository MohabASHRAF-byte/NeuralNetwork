import numpy as np

class ADALINE:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100, precision: int = 3):
        self.weights = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = 0
        self.precision = precision

    def fit(self, training_data, actual_predicted):
        # Initialize weights to zeros
        number_of_samples, number_of_features = training_data.shape
        self.weights = np.zeros(number_of_features)

        for _ in range(self.epochs):
            # Go through each sample and update weights
            for idx, row in enumerate(training_data):
                # Calculate the output (linear sum of inputs and weights)
                linear_output = np.dot(row, self.weights) + self.bias
                prediction_error = actual_predicted[idx] - linear_output

                # Update weights using the gradient descent rule
                self.weights += self.learning_rate * prediction_error * row
                self.bias += self.learning_rate * prediction_error  # Update bias

                # Round weights and bias to the specified precision
                self.weights = np.round(self.weights, self.precision)
                self.bias = round(self.bias, self.precision)

        print("Final Weights:", self.weights)
        print("Final Bias:", self.bias)

    def predict(self, test):
        # Calculate linear output (no activation function)
        linear_output = np.dot(test, self.weights) + self.bias
        return linear_output
