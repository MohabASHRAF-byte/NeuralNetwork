import numpy as np

from ActivationFunction import sigmoid, derivative_sigmoid
from Models.MLP.Layer import Layer


class Mlp:
    def __init__(self, learning_rate, loss_function='mse', epochs=10):
        self.layers: list[Layer] = []
        self.training_layers: list[Layer] = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.activation = sigmoid
        self.deactivation = derivative_sigmoid
        self.loss_function = loss_function

    def add_input_output_layer(self, input_data, output_data, use_bias=False):
        # Generate the input layer
        nxt_layer_size = len(self.layers[0].neurons)
        input_layer = Layer(use_bias=use_bias, next_layer_size=nxt_layer_size)
        input_layer.init_with_values(input_data)
        # Generate the output layer
        output_layer = Layer(use_bias=use_bias)
        output_layer.init_with_values(output_data)
        #
        layer = [input_layer, *self.layers, output_layer]
        return layer

    def lec_example1(self):
        # Add input layer
        input_layer = Layer(False, 2)
        input_layer.init_with_values([0, 0])
        input_layer.neurons[0].next_weights = np.array([0.21, -.4])
        input_layer.neurons[1].next_weights = np.array([0.15, 0.1])
        # Add hidden layer
        hidden_layer1 = Layer(True, 1)
        hidden_layer1.init_with_size(2)
        hidden_layer1.neurons[0].next_weights = [-.2]
        hidden_layer1.neurons[0].bias = -.3
        hidden_layer1.neurons[1].next_weights = [.3]
        hidden_layer1.neurons[1].bias = .25
        # Add output Layer
        output_layer = Layer(True)
        output_layer.init_with_size(1)
        output_layer.neurons[0].bias = -.4
        layer = [input_layer, hidden_layer1, output_layer]
        self.training_layers = layer

    def feed_forward(self):
        for layer_idx, layer in enumerate(self.training_layers):
            if layer_idx == 0:
                continue
            for neuron_idx, neuron in enumerate(layer.neurons):
                val = 0
                for prev_neuron in self.training_layers[layer_idx - 1].neurons:
                    prev_neuron_val = prev_neuron.value
                    edge = prev_neuron.next_weights[neuron_idx]
                    val += prev_neuron_val * edge
                val += neuron.bias
                neuron.value = self.activation(val)

    def back_propagate(self, output_data):
        # Calculate sigma for each layer
        for expected, neuron in zip(output_data, self.training_layers[-1].neurons):
            actual = neuron.value
            neuron.sigma = self.deactivation(actual) * (expected - actual)
        prev_last_idx = len(self.training_layers) - 2
        for i in range(prev_last_idx, -1, -1):
            layer = self.training_layers[i]
            nxt_layer = self.training_layers[i + 1]
            for neuron_idx, neuron in enumerate(layer.neurons):
                sigmas = 0
                for edge, nxt_neuron in zip(neuron.next_weights, nxt_layer.neurons):
                    sigmas += edge * nxt_neuron.sigma
                neuron.sigma = sigmas * self.deactivation(neuron.value)

    def update_weights(self):
        for i in range(len(self.training_layers) - 1):
            layer = self.training_layers[i]
            next_layer = self.training_layers[i + 1]

            for neuron in layer.neurons:
                nw=[]
                for idx, weight in enumerate(neuron.next_weights):
                    nxt_node_sigma = next_layer.neurons[idx].sigma
                    # Update the weight correctly
                    neuron.next_weights[idx] += self.learning_rate * nxt_node_sigma * neuron.value
                    nw.append(neuron.next_weights[idx] )
                neuron.next_weights=nw
    def calculate_loss(self, input_data, output_data):
        total_loss = 0
        for training_input, training_output in zip(input_data, output_data):
            for input_value, neuron in zip(training_input, self.training_layers[0].neurons):
                neuron.value = input_value
            self.feed_forward()

            if self.loss_function == 'mse':
                predicted_output = self.training_layers[-1].neurons[0].value
                # Assuming training_output is a list, extract the first value for MSE calculation
                error = (training_output[0] - predicted_output) ** 2
                total_loss += error
        return total_loss / len(input_data)

    def fit(self, input_data=None, output_data=None):
        # todo loop over epochs for each row in dataset do that
        # self.training_layers = self.add_input_output_layer(input_data, output_data)
        for epoch in range(self.epochs):
            for training_input, training_output in zip(input_data, output_data):
                for input_value, neuron in zip(training_input, self.training_layers[0].neurons):
                    neuron.value = input_value
                self.feed_forward()
                self.back_propagate(training_output)
                self.update_weights()
            loss = self.calculate_loss(input_data, output_data)
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss}')
#                 for i in self.training_layers:
#                     for ii in i.neurons:
#                         print(ii)
#                     print("^" * 50)
#                 print(100*'*')