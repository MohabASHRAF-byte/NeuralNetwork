import numpy as np
import math
def unit_step(x):
    return np.where(x > 0 , 1, 0)

def signum(x):
    return np.where(x > 0, 1, np.where(x < 0, 0, -1))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def derivative_sigmoid(x:float)->float:
    return x * (1 - x)