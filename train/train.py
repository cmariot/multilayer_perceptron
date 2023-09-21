# ************************************************************************** #
#                                                                            #
#                                                       :::      ::::::::    #
#    train.py                                         :+:      :+:    :+:    #
#                                                   +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>          +#+  +:+       +#+         #
#                                               +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:03 by cmariot         #+#    #+#              #
#    Updated: 2023/08/24 14:39:04 by cmariot        ###   ########.fr        #
#                                                                            #
# ************************************************************************** #

from get_datasets import (get_training_data,
                          get_validation_data)
from parse_arguments import parse_arguments
from MultilayerPerceptron import MultilayerPerceptron
from ft_progress import ft_progress
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


def header():
    print("""
              _ _   _     __
  /\\/\\  _   _| | |_(_)   / /  __ _ _   _  ___ _ __
 /    \\| | | | | __| |  / /  / _` | | | |/ _ \\ '__|
/ /\\/\\ \\ |_| | | |_| | / /__| (_| | |_| |  __/ |
\\/    \\/\\__,_|_|\\__|_| \\____/\\__,_|\\__, |\\___|_|
   ___                        _    |___/
  / _ \\___ _ __ ___ ___ _ __ | |_ _ __ ___  _ __
 / /_)/ _ \\ '__/ __/ _ \\ '_ \\| __| '__/ _ \\| '_ \\
/ ___/  __/ | | (_|  __/ |_) | |_| | | (_) | | | |
\\/    \\___|_|  \\___\\___| .__/ \\__|_|  \\___/|_| |_|
                       |_|
""")


# TODO :
# - [ ] Batch the data and use epochs
# - [ ] Use the validation dataset to check and avoid overfitting
# - [ ] Use the model for the training : model.fit(X, y, epochs=100_000)
# - [ ] Use main arguments as model parameters
# - [ ] Fine tuning the hyperparameter default values


if __name__ == "__main__":

    header()

    (
        train_path,       # Path to the training dataset
        validation_path,  # Path to the validation dataset
        n_neurons,        # Number of neurons in each layer
        activations,      # Activation function in each layer
        loss_name,        # Loss function
        epochs,           # Number of epochs
        batch_size,       # Batch size
        learning_rate,    # Initial learning rate
        decay,            # How much the learning rate decreases over time
        momentum          # Avoid local minima and speed up SGD
    ) = parse_arguments()

    # ########################################################### #
    # Load the datasets :                                         #
    # - Training dataset is used to train the model               #
    # - Validation dataset is used to check the model performance #
    #                                                             #
    # The dataset features are normalized (between 0 and 1)       #
    #                                                             #
    # The dataset targets are replaced by 0 for malignant and     #
    # 1 for benign.                                               #
    # ########################################################### #

    (
        x_train_norm,
        y_train,
        x_min,
        x_max
    ) = get_training_data(train_path)

    (
        x_validation_norm,
        y_validation
    ) = get_validation_data(validation_path, x_min, x_max)

    n_features = x_train_norm.shape[1]
    n_train_samples = x_train_norm.shape[0]

    # Model :
    input_layer = Layer(
        n_inputs=30,
        n_neurons=24,
        activation_function="sigmoid",
    )

    hidden_layer1 = Layer(
        n_inputs=24,
        n_neurons=24,
        activation_function="sigmoid",
    )

    hidden_layer2 = Layer(
        n_inputs=24,
        n_neurons=24,
        activation_function="sigmoid",
    )

    ouput_layer = Layer(
        n_inputs=24,
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

    y_train = np.array([[1, 0] if i == 0 else [0, 1] for i in y_train])

    # Metrics :
    losses = []
    accuracies = []
    learning_rates = []

    # Training :
    epochs = 1_000
    for i in ft_progress(range(epochs)):

        # Forwardpropagation :
        input_layer.forward(x_train_norm)
        input_layer.activation_function.forward(input_layer.output)
        hidden_layer1.forward(input_layer.activation_function.output)
        hidden_layer1.activation_function.forward(hidden_layer1.output)
        hidden_layer2.forward(hidden_layer1.activation_function.output)
        hidden_layer2.activation_function.forward(hidden_layer2.output)
        ouput_layer.forward(hidden_layer2.activation_function.output)
        ouput_layer.activation_function.forward(ouput_layer.output)

        # Loss function :
        loss = loss_function.calculate(ouput_layer.activation_function.output, y_train)
        losses.append(loss)

        # Compute and save accuracy :
        y_hat = np.zeros(y_train.shape)
        y_hat[np.arange(len(y_hat)), ouput_layer.activation_function.output.argmax(axis=1)] = 1
        accuracy = accuracy_score_(y_train, y_hat)
        accuracies.append(accuracy)
        learning_rates.append(optimizer.current_learning_rate)

        # Loss backward :
        loss_function.backward(ouput_layer.activation_function.output, y_train)

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

