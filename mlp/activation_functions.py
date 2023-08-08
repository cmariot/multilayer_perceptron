import numpy as np


def linear(x):
    """
    The linear function is used as an activation function.
    It is a linear function.
    """
    return x


def step(x):
    """
    The step function is used as an activation function.
    It is a linear function.
    """
    return np.heaviside(x, 1)


def sigmoid(x):
    """
    The sigmoid function is used as an activation function.
    """
    return 1 / (1 + np.exp(-x))


def hyperboloid_tangent(x):
    """
    The hyperboloid tangent function is used as an activation function.
    """
    return np.tanh(x)


def rectified_linear_unit(x):
    """
    The rectified linear unit function is used as an activation function.
    """
    return np.maximum(0, x)