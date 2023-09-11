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

    print("Number of neurons : ", n_neurons)

    # Create the neural network based on arguments
    model = MultilayerPerceptron(
        n_neurons=n_neurons
    )
