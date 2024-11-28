import random

from prettytable import PrettyTable
from scipy.constants import value

from Models.MLP.Neuron import Neuron
"""
Layer Class Parameters:

1. **use_bias (bool)**:
   - Determines whether bias terms are included in the neurons of this layer.
   - Default: `False`.

2. **next_layer_size (int)**:
   - Specifies the number of neurons in the next layer.
   - Used to initialize weights for connections between this layer and the next layer.
   - Default: `0`.
"""


class Layer:
    def __init__(self, use_bias: bool = False,next_layer_size=0):
        self.neurons: list[Neuron] = []
        self.use_bias = use_bias
        self.next_layer_size= next_layer_size


    def init_with_values(self, values: list):
        for value in values:
            neuron = Neuron(value=value, use_bias=self.use_bias)
            neuron.init_weights(self.next_layer_size)
            self.neurons.append(neuron)

    def init_with_size(self, size: int):
        for i in range(size):
            neuron = Neuron(value= random.random(),use_bias=self.use_bias)
            neuron.init_weights(self.next_layer_size)
            self.neurons.append(neuron)



    def __str__(self):
        table = PrettyTable()
        table.field_names=["Neuron" , "value" , "Sigma","Use_bias"]
        for i,neuron in enumerate(self.neurons):
            table.add_row([f"N{i}" , neuron.value , neuron.sigma,neuron.use_bias])

        return table.get_string()

