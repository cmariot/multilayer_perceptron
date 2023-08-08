import numpy as np
from activation_functions import (
    linear,
    step,
    sigmoid,
    hyperboloid_tangent,
    rectified_linear_unit
)


# Decorator to check the inputs of the perceptron function.
def check_inputs(func):
    def wrapper(inputs, weights, biais, activation_function):
        if len(inputs) != len(weights):
            raise ValueError(
                "Error: the number of inputs and weights must be the same."
            )
        elif activation_function not in [
            linear,
            step,
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


@check_inputs
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
    weighted_sum = np.dot(weights, inputs) + biais
    output = activation_function(weighted_sum)
    return output


if __name__ == "__main__":

    # One input, with a layer of 3 neurons.
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

    # The same result can be obtained with a matrix multiplication.
    weights = np.array([weights1, weights2, weights3])
    biais = np.array([biais1, biais2, biais3])
    perceptron = np.dot(weights, inputs) + biais
    print(perceptron)

    # Now, we have more than one input. With a layer of 3 neurons.
    inputs = np.array([
        [1.2, 1.3, 0.7],
        [2.1, 0.3, 1.7],
        [0.2, 1.3, 2.7],
        [1.2, 1.3, 0.7],
        [2.1, 0.3, 1.7],
        [0.2, 1.3, 2.7],
        [1.2, 1.3, 0.7]
    ])

    weights = np.array([weights1, weights2, weights3])
    biais = np.array([biais1, biais2, biais3])

    try:
        layer = np.dot(inputs, weights.T) + biais
        print(layer)
    except ValueError as error:
        print(error)
        exit(1)

    inputs = np.array([
        [1.2, 1.3, 0.7],
        [2.1, 0.3, 1.7],
        [0.2, 1.3, 2.7],
        [1.2, 1.3, 0.7],
        [2.1, 0.3, 1.7],
        [0.2, 1.3, 2.7],
        [1.2, 1.3, 0.7]
    ])

    # Now, we have more than one layer.
    # The output of the first layer is the input of the second layer.
    weights_l1 = np.array([weights1, weights2, weights3])
    weights_l2 = np.array([weights3, weights2, weights1])

    biais_l1 = np.array([biais1, biais2, biais3])
    biais_l2 = np.array([biais3, biais2, biais1])

    try:
        layer_1_output = np.dot(inputs, weights_l1.T) + biais_l1
        print(layer_1_output)
        layer_2_output = np.dot(layer_1_output, weights_l2.T) + biais_l2
        print(layer_1_output)
    except ValueError as error:
        print(error)
        exit(1)
