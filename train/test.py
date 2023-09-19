import nnfs
from layer import Layer
from nnfs.datasets import spiral_data
from Activation.softmax_categorical_cross_entropy import Softmax_Categorical_Cross_Entropy
import numpy as np


if __name__ == "__main__":

    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer(
        n_inputs=2,
        n_neurons=64,
        activation_function="relu"
    )

    dense2 = Layer(
        n_inputs=64,
        n_neurons=3,
        activation_function="softmax"
    )

    loss_function = Softmax_Categorical_Cross_Entropy()

    losses = []
    for i in range(10001):

        dense1.forward(X)
        dense1.activation_function.forward(dense1.output)
        dense2.forward(dense1.activation_function.output)

        loss = loss_function.forward(dense2.output, y)
        losses.append(loss)

        predictions = np.argmax(loss_function.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        loss_function.backward(loss_function.output, y)
        dense2.backward(loss_function.dinputs)
        dense1.activation_function.backward(dense2.dinputs)
        dense1.backward(dense1.activation_function.dinputs)

        dense2.update()
        dense1.update()


    import matplotlib.pyplot as plt

    plt.plot(range(10001), losses)
    plt.show()


    