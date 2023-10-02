# *************************************************************************** #
#                                                                             #
#                                                        :::      ::::::::    #
#    train.py                                          :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#    By: cmariot <contact@charles-mariot.fr>       +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#    Created: 2023/09/26 13:59:17 by cmariot          #+#    #+#              #
#    Updated: 2023/10/02 11:39:31 by cmariot         ###   ########.fr        #
#                                                                             #
# *************************************************************************** #


import matplotlib.pyplot as plt
from multilayer_perceptron.parse_arguments import parse_arguments
from multilayer_perceptron.get_datasets import (get_training_data,
                                                get_validation_data)
from multilayer_perceptron.MultilayerPerceptron import MultilayerPerceptron
import pandas
import numpy as np


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


def print_metrics(training_metrics, validation_metrics):

    # Dataframe with the last value of the training and validation metrics
    metrics_dataframe = {
        "Training": [
            training_metrics["loss"][-1],
            training_metrics["accuracy"][-1],
            training_metrics["recall"][-1],
            training_metrics["precision"][-1],
            training_metrics["f1_score"][-1]
        ],
        "Validation": [
            validation_metrics["loss"][-1],
            validation_metrics["accuracy"][-1],
            validation_metrics["recall"][-1],
            validation_metrics["precision"][-1],
            validation_metrics["f1_score"][-1]
        ]
    }
    # Print the dataframe
    print("\n", pandas.DataFrame(metrics_dataframe, index=[
        "Loss",
        "Accuracy",
        "Recall",
        "Precision",
        "F1 score"
    ]).to_string())


def plot_metrics(training_metrics, validation_metrics):
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    for metric, ax in zip(training_metrics, axs.flat):
        ax.set_title(metric)
        ax.plot(training_metrics[metric], label="Training")
        ax.plot(validation_metrics[metric], label="Validation")
        ax.legend()
        ax.set_ylim([0, 1.1])
    plt.legend()
    plt.show()


def plot_loss(training_metrics, validation_metrics):
    plt.title("Loss evolution computed on training and validation datasets")
    plt.plot(training_metrics["loss"], label="Training loss")
    plt.plot(validation_metrics["loss"], label="Validation loss")
    plt.legend()
    plt.show()


# def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     """
#     plt.figure()
#     plt.imshow(cm, interpolation="nearest", cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, ha="right")
#     plt.yticks(tick_marks, classes)
#     # Put the values inside the confusion matrix
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             # Put the values inside the confusion matrix
#             value = cm.iloc[i, j]
#             plt.text(
#                 j,
#                 i,
#                 format(cm.iloc[i, j], "d"),
#                 ha="center",
#                 va="center",
#                 color="white" if value > 100 else "black"
#             )
#     plt.tight_layout()
#     plt.ylabel("True label")
#     plt.xlabel("Predicted label")
#     plt.show()


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
        training_set,
        x_min,
        x_max
    ) = get_training_data(train_path)

    (
        x_validation_norm,
        y_validation
    ) = get_validation_data(validation_path, x_min, x_max)

    # ########################################################### #
    # Create the neural network model :                           #
    # - Input layer                                               #
    # - Hidden layer 1                                            #
    # - Hidden layer 2                                            #
    # - Output layer                                              #
    # ########################################################### #

    # Create the multilayer perceptron object
    model = MultilayerPerceptron(
        n_neurons=n_neurons,
        activations=activations,
        loss_name=loss_name,
        epochs=epochs,
        batch_size=batch_size,
        n_train_samples=len(x_train_norm),
        learning_rate=learning_rate,
        x_min=x_min,
        x_max=x_max,
    )

    # ############### #
    # Train the model #
    # ############### #

    model.fit(
        training_set,
        x_train_norm,
        y_train,
        x_validation_norm,
        y_validation
    )

    model.save_model("../model.pkl")

    # ############### #
    # Test the model  #
    # ############### #

    print_metrics(model.training_metrics, model.validation_metrics)

    # Plot the loss of the training and validation sets on the same graph
    plot_loss(model.training_metrics, model.validation_metrics)

    # Plot the confusion matrix on the validation set
    # plot_confusion_matrix(
    #     confusion_matrix_(y_validation, validation_y_hat, df_option=True),
    #     classes=["Malignant", "Benign"],
    #     title="Confusion matrix on the training set"
    # )

    plot_metrics(model.training_metrics, model.validation_metrics)
