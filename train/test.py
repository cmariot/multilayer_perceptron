import nnfs
from layer import Layer
from nnfs.datasets import spiral_data
import numpy as np
from ft_progress import ft_progress
from model import Model
from Metrics.accuracy import accuracy_score_
import matplotlib.pyplot as plt
from Optimizers.sgd import StandardGradientDescent
from Loss.binary_cross_entropy import BinaryCrossEntropy_Loss


# TODO :
# - [ ] Replace the categorical by Softmax activation function and the Binary Cross Entropy Loss
# - [ ] Add hidden layers
# - [ ] Use the model for the training : model.fit(X, y, epochs=100_000)
# - [ ] Batch the data and use epochs


if __name__ == "__main__":

    # Dataset :
    X, y = spiral_data(samples=100, classes=2)
    y = y.reshape(-1, 1)

    # Model :
    dense1 = Layer(
        n_inputs=2,
        n_neurons=64,
        activation_function="sigmoid",
    )

    dense2 = Layer(
        n_inputs=64,
        n_neurons=2,
        activation_function="softmax",
    )

    optimizer = StandardGradientDescent(
        learning_rate=0.5,
        decay=1e-3,
        momentum=0.95,
    )

    loss_function = BinaryCrossEntropy_Loss()

    model = Model([dense1, dense2])

    # Metrics :
    losses = []
    accuracies = []
    learning_rates = []

    # Training :
    epochs = 10 # 50_000
    for i in ft_progress(range(epochs)):

        print("Epoch: ", i, "\n\n")

        # Forwardpropagation :
        # Input layer
        dense1.forward(X)
        dense1.activation_function.forward(dense1.output)
        # Hidden layer
        # TODO [...]
        # Output layer
        dense2.forward(dense1.activation_function.output)
        dense2.activation_function.forward(dense2.output)

        # Loss function :
        loss = loss_function.calculate(dense2.activation_function.output, y)
        losses.append(loss)

        # Compute and save accuracy :
        y_hat = np.argmax(dense2.activation_function.output, axis=1)
        accuracy = accuracy_score_(y, y_hat)
        accuracies.append(accuracy)
        learning_rates.append(optimizer.current_learning_rate)

        # # Backpropagation :
        # # Input layer
        # loss_function.backward(dense2.activation_function.output, y)
        # dense2.backward(loss_function.dinputs)
        # # Hidden layer
        # # TODO [...]
        # # Output layer
        # dense1.activation_function.backward(dense2.dinputs)
        # dense1.backward(dense1.activation_function.dinputs)

        # # Update weights and biases :
        # optimizer.pre_update_params()
        # optimizer.update(dense1)
        # optimizer.update(dense2)
        # optimizer.post_update_params()

    # Print the last value of loss and accuracy
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    # Plot the loss and accuracy on the same graph
    plt.title("Loss and Accuracy")
    plt.plot(range(epochs), losses, label="loss")
    plt.plot(range(epochs), accuracies, label="accuracy")
    plt.legend()
    plt.show()

    # Plot the loearning rate evolution
    plt.title("Learning Rate Decay")
    plt.plot(range(epochs), learning_rates)
    plt.show()

