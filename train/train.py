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


import numpy as np
import matplotlib.pyplot as plt
from parse_arguments import parse_arguments
from get_datasets import (get_training_data,
                          get_validation_data)
from MultilayerPerceptron import MultilayerPerceptron
from ft_progress import ft_progress
from Metrics.accuracy import accuracy_score_
from Metrics.precision import precision_score_
from Metrics.recall import recall_score_
from Metrics.f1_score import f1_score_
from Metrics.confusion_matrix import confusion_matrix_


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


def get_batch(x_train, y_train, i, batch_size):
    """
    """
    try:
        n_samples = x_train.shape[0]
        train_set = np.concatenate((x_train, y_train), axis=1)
        index_start = np.random.randint(0, n_samples)
        index_end = index_start + batch_size
        if index_end > n_samples:
            batch_begin = train_set[index_start:, :]
            batch_end = train_set[:index_end - n_samples, :]
            batch = np.concatenate((batch_begin, batch_end), axis=0)
        else:
            batch = train_set[index_start:index_end, :]
        x = batch[:, :-2]
        y = batch[:, -2:]
        return x, y
    except Exception as error:
        print(error)
        exit()


def metrics_dictionary():
    """
    Return a dictionary with the metrics as keys and empty lists as values.
    Used to store the metrics that will be plotted.
    """
    try:
        return {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "loss": []
        }
    except Exception as error:
        print(error)
        exit()


def compute_metrics(model, x, y, dictionary):
    try:
        output = model.forward(x)
        y_hat = np.zeros(y.shape)
        y_hat[np.arange(len(y_hat)), output.argmax(axis=1)] = 1
        dictionary["loss"].append(model.loss.calculate(output, y))
        dictionary["accuracy"].append(accuracy_score_(y, y_hat))
        dictionary["precision"].append(precision_score_(y, y_hat))
        dictionary["recall"].append(recall_score_(y, y_hat))
        dictionary["f1_score"].append(f1_score_(y, y_hat))
        return dictionary, y_hat
    except Exception as error:
        print(error)
        exit()


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

    n_train_samples = x_train_norm.shape[0]
    n_features = x_train_norm.shape[1]

    y_train = np.array([[1, 0] if i == 0 else [0, 1] for i in y_train])
    y_validation = np.array([[1, 0] if i == 0 else [0, 1] for i in y_validation])

    # ########################################################### #
    # Create the neural network model :                           #
    # - Input layer                                               #
    # - Hidden layer 1                                            #
    # - Hidden layer 2                                            #
    # - Output layer                                              #
    # ########################################################### #

    # Create the multilayer perceptron object
    model = MultilayerPerceptron(
        n_features=n_features,
        n_neurons=n_neurons,
        activations=activations,
        loss_name=loss_name,
        epochs=epochs,
        batch_size=batch_size,
        n_train_samples=n_train_samples,
        learning_rate=learning_rate,
        decay=decay,
        momentum=momentum
    )

    # ############### #
    # Train the model #
    # ############### #

    # Metrics :
    learning_rates = []
    training_metrics = metrics_dictionary()
    validation_metrics = metrics_dictionary()

    # Training :
    for epoch in ft_progress(range(model.epochs)):
        for i in range(model.n_batch):
            x_batch, y_batch = get_batch(
                x_train_norm, y_train, i, model.batch_size
            )
            y_hat = model.forward(x_batch)
            model.backward(y_batch)
            learning_rates.append(model.optimizer.current_learning_rate)
            model.optimize()

        # ############### #
        # Train the model #
        # ############### #

        # Compute the metrics on the training dataset :
        training_metrics, train_y_hat = compute_metrics(
            model,
            x_train_norm,
            y_train,
            training_metrics
        )

        # Compute the metrics on the validation dataset :
        validation_metrics, validation_y_hat = compute_metrics(
            model,
            x_validation_norm,
            y_validation,
            validation_metrics
        )


    # Print the last value of loss and accuracy
    print("Loss: ", training_metrics["loss"][-1])
    print("Accuracy: ", training_metrics["accuracy"][-1])

    # Confusion matrix :
    print("\nConfusion matrix on the training set:\n\n")
    confusion_matrix_(y_train, train_y_hat, df_option=True)
    print("\nConfusion matrix on the validation set:\n\n")
    confusion_matrix_(y_validation, validation_y_hat, df_option=True)

    # Plot the loss and accuracy on the same graph
    plt.title("Loss evolution computed on training and validation datasets")
    plt.plot(training_metrics["loss"], label="Training loss")
    plt.plot(validation_metrics["loss"], label="Validation loss")
    plt.legend()
    plt.show()

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    for metric, ax in zip(training_metrics, axs.flat):
        ax.set_title(metric)
        ax.plot(training_metrics[metric], label="Training")
        ax.plot(validation_metrics[metric], label="Validation")
        ax.legend()
    plt.legend()
    plt.show()

    # Plot the learning rate evolution
    plt.title("Learning Rate Decay")
    plt.plot(learning_rates)
    plt.show()

