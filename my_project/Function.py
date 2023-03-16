import numpy as np
from scipy.special import expit


def logistic_function(x):
    return expit(x)


def generate_random_array(shape):
    return np.random.rand(*shape)
