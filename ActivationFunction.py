import numpy as np
def unit_step(x):
    return np.where(x > 0 , 1, 0)

def signum(x):

    return np.where(x > 0, 1, np.where(x < 0, 0, -1))
