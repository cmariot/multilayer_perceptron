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
from get_datasets import (get_training_data,
                          get_validation_data)
from parse_arguments import parse_arguments
from multi_layer_perceptron import MultiLayerPerceptron
from ft_progress import ft_progress
from Metrics.compute import (metrics_dictionary,
                             get_batch,
                             compute_metrics,
                             print_final_metrics)
from Metrics.confusion_matrix import confusion_matrix_
from Plots.loss import plot_loss
from Plots.metrics import plot_metrics
from Plots.learning_rate import plot_learning_rate


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

    # ################################################# #
    # Create the neural network :                       #
    # - Input layer    : sigmoid 30 inputs / 30 outputs #
    # - Hidden layer 1 : sigmoid 30 inputs / 24 outputs #
    # - Hidden layer 2 : sigmoid 24 inputs / 24 outputs #
    # - Hidden layer 3 : sigmoid 24 inputs / 24 outputs #
    # - Output layer   : softmax 24 inputs /  2 outputs #
    # ################################################# #

    # TODO:
    # - I need to check if everything is ok.
    #       - Loss function : BinaryCrossEntropy (forward)
    #       - Gradient : BinaryCrossEntropy (backward)
    #       - Backward pass
    #       - Decay check
    #       - Momentum check
    # - Loss + Activation output in the same class ?
    # - Fine tune the hyperparameters :
    #   - Number of epochs
    #   - Initial learning rate
    #   - Decay
    #   - Momentum
    #   - Batch size
    #   - Number of neurons in each layer
    # - Save the model
    # - Load the model (weights and biases)
    # - Predict

    multilayer_perceptron = MultiLayerPerceptron(
        n_features=n_features,            # Number of inputs in the first layer
        n_neurons=n_neurons,              # Number of outputs in each layer
        activations=activations,          # Activation function in each layer
        learning_rate=learning_rate,      # Learning rate
        decay=decay,                      # Learning_rate decreases over time
        momentum=momentum,                # Avoid local minima and speed up SGD
        batch_size=batch_size,            # Batch size
        n_train_samples=n_train_samples,  # Training set size
        loss_name=loss_name               # Loss function
    )

    # ##################################### #
    # Train the neural network              #
    # ##################################### #

    # Variables used to save the metrics for the plots
    losses_training = []
    losses_validation = []
    training_metrics = metrics_dictionary()
    validation_metrics = metrics_dictionary()

    input("Press enter to train the model...\n")

    for epoch in ft_progress(range(epochs)):

        batch_losses = []
        batch_train_metrics = metrics_dictionary()
        batch_validation_metrics = metrics_dictionary()

        for i in range(multilayer_perceptron.n_batch):

            x_batch, y_batch = get_batch(x_train_norm, y_train, batch_size)

            # Forward pass
            output = multilayer_perceptron.forward(x_batch)
            y_hat = multilayer_perceptron.predict(output)

            loss = multilayer_perceptron.loss(y_hat, y_batch)

            # Compute metrics on the training set
            batch_losses.append(loss)
            compute_metrics(batch_train_metrics, y_batch, y_hat)

            # Backward pass
            gradient = multilayer_perceptron.gradient(output, y_batch)
            multilayer_perceptron.backward(gradient)
            multilayer_perceptron.gradient_descent()

        multilayer_perceptron.update_learning_rate()

        # #################### #
        # Training set metrics #
        # #################### #

        losses_training.append(np.mean(batch_losses))
        for i, (metric, list_) in enumerate(batch_train_metrics.items()):
            training_metrics[metric].append(np.mean(list_))

        # ##################################### #
        # Validation set metrics :              #
        # Compute metrics on the validation set #
        # ##################################### #

        output = multilayer_perceptron.forward(x_validation_norm)
        y_hat = multilayer_perceptron.predict(output)
        loss = multilayer_perceptron.loss(y_hat, y_validation)
        losses_validation.append(loss)
        compute_metrics(validation_metrics, y_validation, y_hat)

    # ###################################### #
    # Confusion Matrix on the validation set #
    # ###################################### #

    confusion_matrix_(
        y_true=y_validation,
        y_hat=y_hat,
        labels=["Malignant", "Benign"],
        df_option=True
    )

    # ############################### #
    # Final metrics on validation set #
    # ############################### #

    print_final_metrics(validation_metrics)

    # ##################################### #
    # Plots                                 #
    # ##################################### #

    plot_loss(losses_training, losses_validation)
    plot_metrics(training_metrics, validation_metrics)
    plot_learning_rate(multilayer_perceptron.learning_rates)

    # ##################################### #
    # Save the model                        #
    # ##################################### #

    # TODO:
    # - Save the model (weights, biases)
    # - Bonus: Different opimizers
