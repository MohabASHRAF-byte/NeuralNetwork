import numpy as np
from numpy.random import random


class Neuron:
    def __init__(self, value: float, use_bias: bool, bias: float = None):
        self.bias = random() if use_bias else 0
        self.bias = bias if bias is not None else self.bias
        self.use_bias = use_bias
        self.sigma = 0
        self.value = value
        self.next_weights = None
    def init_weights(self,sz):
        self.next_weights= np.random.random(sz)

    def __str__(self):
        output=""
        output += f"Value = {self.value}\n"
        output += f"Sigma = {self.sigma}\n"
        output += f"use_bias = {self.use_bias}\n"
        output += f"Weights = {self.next_weights}\n"
        output += f"Bias = {self.bias}\n"
        return output