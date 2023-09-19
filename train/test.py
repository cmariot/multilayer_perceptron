import nnfs
from layer import Layer
from nnfs.datasets import spiral_data
from Activation.softmax_categorical_cross_entropy import Softmax_Categorical_Cross_Entropy
import numpy as np
from ft_progress import ft_progress
from model import Model


if __name__ == "__main__":

    X, y = spiral_data(samples=100, classes=2)

    dense1 = Layer(
        n_inputs=2,
        n_neurons=64,
        activation_function="sigmoid",
        optimizer="sgd",
        learning_rate=0.1,
        decay=1e-3,
        momentum=0.95,
    )

    dense2 = Layer(
        n_inputs=64,
        n_neurons=3,
        activation_function="softmax",
        optimizer="sgd",
        learning_rate=0.1,
        decay=1e-3,
        momentum=0.95,
    )

    model = Model([dense1, dense2])

    loss_function = Softmax_Categorical_Cross_Entropy()

    losses = []
    accuracies = []

    learning_rates = []

    optimizer = dense1.optimizer

    epochs = 1_000_000
    for i in ft_progress(range(epochs)):

        dense1.forward(X)
        dense1.activation_function.forward(dense1.output)
        dense2.forward(dense1.activation_function.output)

        # Activation + Loss
        loss = loss_function.forward(dense2.output, y)
        losses.append(loss)

        predictions = np.argmax(loss_function.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        accuracies.append(accuracy)

        loss_function.backward(loss_function.output, y)
        dense2.backward(loss_function.dinputs)
        dense1.activation_function.backward(dense2.dinputs)
        dense1.backward(dense1.activation_function.dinputs)

        optimizer.pre_update_params()
        optimizer.update(dense1)
        optimizer.update(dense2)
        optimizer.post_update_params()

        learning_rates.append(dense1.optimizer.current_learning_rate)


    # Print the last 5 values of loss and accuracy
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


    import matplotlib.pyplot as plt

    plt.title("Loss and Accuracy")
    plt.plot(range(epochs), losses, label="loss")
    plt.plot(range(epochs), accuracies, label="accuracy")
    plt.legend()
    plt.show()

    plt.title("Learning Rate Decay")
    plt.plot(range(epochs), learning_rates)
    plt.show()


    