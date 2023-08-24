# ****************************************************************************#
#                                                                             #
#                                                         :::      ::::::::   #
#    train.py                                           :+:      :+:    :+:   #
#                                                     +:+ +:+         +:+     #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+        #
#                                                 +#+#+#+#+#+   +#+           #
#    Created: 2023/08/24 14:39:03 by cmariot           #+#    #+#             #
#    Updated: 2023/08/24 14:39:04 by cmariot          ###   ########.fr       #
#                                                                             #
# ****************************************************************************#

from os import get_terminal_size
from time import time
import pandas
import numpy as np
from get_datasets import (get_training_data,
                          get_validation_data,
                          dataset_loaded_message)
from parse_arguments import parse_arguments
from Loss.binary_cross_entropy import BinaryCrossEntropy_Loss
from Plots.loss import plot_loss
from Plots.metrics import plot_metrics
from Plots.learning_rate import plot_learning_rate
from Metrics.accuracy import accuracy_score_
from Metrics.f1_score import f1_score_
from Metrics.precision import precision_score_
from Metrics.recall import recall_score_
from Metrics.confusion_matrix import confusion_matrix_
from multi_layer_perceptron import MultiLayerPerceptron


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


def metrics_dict():
    return {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }


def get_batch(x, y, i, batch_size):
    start = i * batch_size
    end = (i + 1) * batch_size
    if end > x.shape[0]:
        end = x.shape[0]
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def ft_progress(iterable,
                length=get_terminal_size().columns - 4,
                fill='█',
                empty='░',
                print_end='\r'):
    """
    Progress bar generator.
    """

    def get_elapsed_time_str(elapsed_time):
        """
        Return the elapsed time as str.
        """
        if elapsed_time < 60:
            return f'[Elapsed-time {elapsed_time:.2f} s]'
        elif elapsed_time < 3600:
            return f'[Elapsed-time {elapsed_time / 60:.0f} m]'
        else:
            return f'[Elapsed-time {elapsed_time / 3600:.0f} h]'

    def get_eta_str(eta):
        """
        Return the Estimed Time Arrival as str.
        """
        if eta == 0.0:
            return ' [DONE]                         '
        elif eta < 60:
            return f' [{eta:.0f} s remaining]       '
        elif eta < 3600:
            return f' [{eta / 60:.0f} m remaining]  '
        else:
            return f' [{eta / 3600:.0f} h remaining]'

    try:
        print()
        total = len(iterable)
        start = time()
        for i, item in enumerate(iterable, start=1):
            elapsed_time = time() - start
            et_str = get_elapsed_time_str(elapsed_time)
            eta_str = get_eta_str(elapsed_time * (total / i - 1))
            filled_length = int(length * i / total)
            percent_str = f'[{(i / total) * 100:6.2f} %] '
            progress_str = str(fill * filled_length
                               + empty * (length - filled_length))
            counter_str = f'  [{i:>{len(str(total))}}/{total}] '
            bar = ("\033[F\033[K  " + progress_str + "\n"
                   + counter_str
                   + percent_str
                   + et_str
                   + eta_str)
            print(bar, end=print_end)
            yield item
        print()
    except Exception:
        print("Error: ft_progress")
        return None


if __name__ == "__main__":

    header()

    (
        layers,           # Number of outputs in each layer
        activations,      # Activation function in each layer
        loss_name,        # Loss function
        epochs,           # Number of epochs
        batch_size,       # Batch size
        learning_rate     # Learning rate
    ) = parse_arguments()

    # ############################################### #
    # Load the datasets,                              #
    # - Training dataset is used to train the model   #
    # - Validation dataset is used to check the model #
    #                                                 #
    # The dataset features are normalized             #
    # (between 0 and 1)                               #
    #                                                 #
    # The dataset targets are replaced by 0 for       #
    #  Malignant and 1 for benign.                    #
    # ############################################### #

    (
        x_train_norm,
        y_train,
        x_min,
        x_max
    ) = get_training_data("../datasets/train.csv")

    (
        x_validation_norm,
        y_validation
    ) = get_validation_data("../datasets/validation.csv", x_min, x_max)

    (
        n_features,
        n_train_samples
    ) = dataset_loaded_message(x_train_norm, x_validation_norm)

    # Press enter to continue
    input("Press enter to create the neural network ...\n")

    # ################################################# #
    # Create the neural network :                       #
    # - Input layer    : sigmoid 30 inputs / 30 outputs #
    # - Hidden layer 1 : sigmoid 30 inputs / 24 outputs #
    # - Hidden layer 2 : sigmoid 24 inputs / 24 outputs #
    # - Hidden layer 3 : sigmoid 24 inputs / 24 outputs #
    # - Output layer   : softmax 24 inputs /  2 outputs #
    # ################################################# #

    multilayer_perceptron = MultiLayerPerceptron(
        n_features=n_features,
        layers=layers,
        activations=activations,
        learning_rate=learning_rate,
        decay=0.0005,   # Decay : learning_rate decreases over time
        momentum=0.05,  # Momentum : avoid local minima and speed up SGD
        batch_size=batch_size,
        n_train_samples=n_train_samples
    )

    loss_function = BinaryCrossEntropy_Loss()

    # ##################################### #
    # Train the neural network              #
    # ##################################### #

    input("Press enter to train the model...\n")

    metrics_functions = [
        accuracy_score_,
        precision_score_,
        recall_score_,
        f1_score_
    ]

    losses_training = []
    losses_validation = []
    learning_rates = []
    training_metrics = metrics_dict()
    validation_metrics = metrics_dict()

    for epoch in ft_progress(range(epochs)):

        batch_losses = []

        batch_train_metrics = metrics_dict()
        batch_validation_metrics = metrics_dict()

        for i in range(multilayer_perceptron.n_batch):

            x_batch, y_batch = get_batch(
                x_train_norm, y_train, i, batch_size
            )

            # Forward pass
            last_layer_output = multilayer_perceptron.forward(x_batch)

            # Get predictions
            y_pred = np.argmax(last_layer_output, axis=1).reshape(-1, 1)

            # Compute the loss
            loss = loss_function.forward(y_pred, y_batch)

            # Compute metrics on the training set
            for i, (metric, list_) in enumerate(batch_train_metrics.items()):
                list_.append(metrics_functions[i](y_batch, y_pred))

            # Save the current loss, used for the plot
            batch_losses.append(loss)

            # calculating the derivative of cost with respect to some weight
            dcost = loss_function.gradient(last_layer_output, y_batch)

            # Backpropagation
            multilayer_perceptron.backward(dcost)

            # Update the learning rate
            multilayer_perceptron.update_learning_rate()

            # Update the weights and the biases
            multilayer_perceptron.update_parameters()

            # Update the iterations
            multilayer_perceptron.update_iterations()

        # #################### #
        # Training set metrics #
        # #################### #

        # Save the training set loss mean for the current epoch
        losses_training.append(np.mean(batch_losses))

        # Append the batch metrics mean to training_metrics
        for i, (metric, list_) in enumerate(batch_train_metrics.items()):
            training_metrics[metric].append(np.mean(list_))

        # ############# #
        # Learning rate #
        # ############# #

        # Save the current learning rate
        learning_rates.append(
            multilayer_perceptron.layers[0].current_learning_rate
        )

        # ###################### #
        # Validation set metrics #
        # ###################### #

        # Compute metrics on the validation set
        last_layer_output = multilayer_perceptron.forward(
            x_validation_norm
        )
        y_pred = np.argmax(last_layer_output, axis=1).reshape(-1, 1)

        # Compute the loss for the validation set
        losses_validation.append(
            loss_function.forward(y_pred, y_validation)
        )

        # Compute other metrics on the validation set
        for i, (metric, list_) in enumerate(validation_metrics.items()):
            list_.append(metrics_functions[i](y_validation, y_pred))

        # TODO:
        # - Activation / Loss backward check
        # - Loss + Activation output in the same class ?

    # ############################### #
    # Final metrics on validation set #
    # ############################### #

    final_validation_metrics = {
        "accuracy": validation_metrics["accuracy"][-1],
        "precision": validation_metrics["precision"][-1],
        "recall": validation_metrics["recall"][-1],
        "f1_score": validation_metrics["f1_score"][-1],
    }

    # Print the final metrics
    df_metrics = pandas.DataFrame(
        final_validation_metrics,
        index=["Validation set metrics"]
    )
    print("\n", df_metrics)

    # ###################################### #
    # Confusion Matrix on the validation set #
    # ###################################### #

    confusion_matrix_(
        y_true=y_validation,
        y_hat=y_pred,
        labels=["Malignant", "Benign"],
        df_option=True
    )

    # ##################################### #
    # Plots                                 #
    # ##################################### #

    # Loss evolution
    plot_loss(losses_training, losses_validation)

    # Accuracy evolution
    plot_metrics(
        training_metrics=training_metrics,
        validation_metrics=validation_metrics
    )

    # Plot the learning rate evolution
    plot_learning_rate(learning_rates)
