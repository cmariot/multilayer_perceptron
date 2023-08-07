import numpy as np


def linear(x):
    """
    The linear function is used as an activation function.
    It is a linear function.
    """
    return x


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


# Decorator to check the inputs of the perceptron function.
def check_inputs(func):
    def wrapper(inputs, weights, biais, activation_function):
        if len(inputs) != len(weights):
            raise ValueError(
                "Error: the number of inputs and weights must be the same."
            )
        elif activation_function not in [
            linear,
            sigmoid,
            hyperboloid_tangent,
            rectified_linear_unit
        ]:
            raise ValueError(
                "Error: the activation function is not valid."
            )
        return func(inputs, weights, biais, activation_function)
    return wrapper


@check_inputs
def iterative_perceptron(inputs, weights, biais, activation_function):
    """
    Each perceptron of the neural network has an input, a weight and a biais.
    The inputs are the features of the dataset, or the outputs of the previous
    layer.
    The weights are the parameters of the model.
    The biais is a constant.
    The output is the sum of the inputs multiplied by the weights, plus the
    biais.
    """
    weighted_sum = biais
    for input, weight in zip(inputs, weights):
        weighted_sum += input * weight
    output = activation_function(weighted_sum)
    return output


def perceptron(inputs, weights, biais, activation_function):
    """
    Each perceptron of the neural network has an input, a weight and a biais.
    The inputs are the features of the dataset, or the outputs of the previous
    layer.
    The weights are the parameters of the model.
    The biais is a constant.
    The output is the sum of the inputs multiplied by the weights, plus the
    biais.
    """
    weighted_sum = np.dot(inputs, weights) + biais
    output = activation_function(weighted_sum)
    return output


if __name__ == "__main__":

    inputs = np.array([1.2, 1.3, 0.7])

    weights1 = np.array([3.1, 2.1, 8.7])
    weights2 = np.array([2.3, 4.1, 0.6])
    weights3 = np.array([5.8, 7.9, 3.4])

    biais1 = 3
    biais2 = 42
    biais3 = 3.5

    try:
        layer = (
            perceptron(inputs, weights1, biais1, linear),
            perceptron(inputs, weights2, biais2, linear),
            perceptron(inputs, weights3, biais3, linear)
        )
        print(layer)
    except ValueError as error:
        print(error)
        exit(1)
