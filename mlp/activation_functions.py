import numpy as np


def linear(weighted_sum):
    """
    The linear function is used as an activation function.
    It is a linear function.
    """
    return weighted_sum


def step(weighted_sum):
    """
    The step function is used as an activation function.
    It is a linear function.
    """
    return np.heaviside(weighted_sum, 1)


def sigmoid(weighted_sum):
    """
    The sigmoid function is used as an activation function.
    """
    return 1 / (1 + np.exp(-weighted_sum))


def hyperboloid_tangent(weighted_sum):
    """
    The hyperboloid tangent function is used as an activation function.
    """
    return np.tanh(weighted_sum)


def activation_ReLU(weighted_sum):
    """
    The rectified linear unit function is used as an activation function.
    """
    output = np.maximum(0, weighted_sum)
    return output


def softmax(weighted_sum):
    """
    The softmax function is used as an activation function.
    """
    exp_values = np.exp(
        weighted_sum -
        np.max(
            weighted_sum,
            axis=1,
            keepdims=True
        )
    )
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities


activation_functions = {
    "linear": linear,
    "step": step,
    "sigmoid": sigmoid,
    "hyperboloid_tangent": hyperboloid_tangent,
    "relu": activation_ReLU,
    "softmax": softmax
}
