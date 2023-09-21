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
from Loss.categorical_cross_entropy import CategoricalCrossEntropy_Loss


# TODO :
# - [ ] Use the real training dataset
# - [ ] Batch the data and use epochs
# - [ ] Use the model for the training : model.fit(X, y, epochs=100_000)
# - [ ] Use main arguments as model parameters
# - [ ] Use the test dataset to check and avoid overfitting
# - [ ] Fine tuning the hyperparameter default values

if __name__ == "__main__":

    # Dataset :
    X, y = spiral_data(samples=300, classes=2)
    y = np.array([[1, 0] if i == 0 else [0, 1] for i in y])

    # Model :
    input_layer = Layer(
        n_inputs=2,
        n_neurons=10,
        activation_function="sigmoid",
    )

    hidden_layer1 = Layer(
        n_inputs=10,
        n_neurons=10,
        activation_function="sigmoid",
    )

    hidden_layer2 = Layer(
        n_inputs=10,
        n_neurons=10,
        activation_function="sigmoid",
    )

    ouput_layer = Layer(
        n_inputs=10,
        n_neurons=2,
        activation_function="softmax",
    )

    optimizer = StandardGradientDescent(
        learning_rate=0.1,
        decay=1e-3,
        momentum=0.99,
    )

    loss_function = BinaryCrossEntropy_Loss()

    model = Model([input_layer, ouput_layer])

    # Metrics :
    losses = []
    accuracies = []
    learning_rates = []

    # Training :
    epochs = 10_000
    for i in ft_progress(range(epochs)):

        # Forwardpropagation :
        input_layer.forward(X)
        input_layer.activation_function.forward(input_layer.output)
        hidden_layer1.forward(input_layer.activation_function.output)
        hidden_layer1.activation_function.forward(hidden_layer1.output)
        hidden_layer2.forward(hidden_layer1.activation_function.output)
        hidden_layer2.activation_function.forward(hidden_layer2.output)
        ouput_layer.forward(hidden_layer2.activation_function.output)
        ouput_layer.activation_function.forward(ouput_layer.output)

        # Loss function :
        loss = loss_function.calculate(ouput_layer.activation_function.output, y)
        losses.append(loss)

        # Compute and save accuracy :
        y_hat = np.zeros(y.shape)
        y_hat[np.arange(len(y_hat)), ouput_layer.activation_function.output.argmax(axis=1)] = 1
        accuracy = accuracy_score_(y, y_hat)
        accuracies.append(accuracy)
        learning_rates.append(optimizer.current_learning_rate)

        # Loss backward :
        loss_function.backward(ouput_layer.activation_function.output, y)

        # Backpropagation :
        ouput_layer.activation_function.backward(loss_function.dinputs)
        ouput_layer.backward(ouput_layer.activation_function.dinputs)

        hidden_layer2.activation_function.backward(ouput_layer.dinputs)
        hidden_layer2.backward(hidden_layer2.activation_function.dinputs)

        hidden_layer1.activation_function.backward(ouput_layer.dinputs)
        hidden_layer1.backward(hidden_layer1.activation_function.dinputs)

        input_layer.activation_function.backward(hidden_layer1.dinputs)
        input_layer.backward(input_layer.activation_function.dinputs)

        # Update weights and biases :
        optimizer.pre_update_params()
        optimizer.update(input_layer)
        optimizer.update(ouput_layer)
        optimizer.post_update_params()

    # Print the last value of loss and accuracy
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    # Plot the loss and accuracy on the same graph
    plt.title("Loss and Accuracy")
    plt.plot(range(epochs), losses, label="loss")
    plt.plot(range(epochs), accuracies, label="accuracy")
    plt.legend()
    plt.show()

    # Plot the learning rate evolution
    # plt.title("Learning Rate Decay")
    # plt.plot(range(epochs), learning_rates)
    # plt.show()

