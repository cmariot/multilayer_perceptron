# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 14:39:03 by cmariot           #+#    #+#              #
#    Updated: 2023/08/24 14:39:04 by cmariot          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from os import get_terminal_size
from time import time
import pandas
import numpy as np
from parse_arguments import parse_args
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
from layer import Dense_Layer


def load_dataset(path):
    try:
        dataset = pandas.read_csv(path)
        return dataset
    except Exception:
        print(f"Error: can't load the dataset {path}")
        exit()


def get_training_data(dataset_path):
    """
    Load the training dataset, normalize the features and
    replace the target labels by 0/1.
    """

    # Load the training dataset
    train_data = load_dataset(dataset_path)

    def normalize_train(x_train):
        """
        Normalize the features of the training dataset.
        All the features will be between 0 and 1.
        Return the normalized features, the minimum and the maximum
        values of each feature, used to normalize the validation set.
        """
        x_min = x_train.min()
        x_max = x_train.max()
        x_train_norm = (x_train - x_min) / (x_max - x_min)
        return x_train_norm.to_numpy(), x_min, x_max

    try:

        # Get the features and normalize them
        x_train = train_data.drop("Diagnosis", axis=1)
        x_train_norm, x_min, x_max = normalize_train(x_train)

        # Get the target and replace the labels by 0/1
        y_train = train_data["Diagnosis"]
        y_train = np.where(y_train == "M", 0, 1)
        y_train = y_train.reshape(-1, 1)

        return x_train_norm, y_train, x_min, x_max

    except Exception:
        print("Error: can't get the training data.")
        exit()


def get_validation_data(dataset_path, x_min, x_max):
    """
    Load the validation dataset, normalize the features and
    replace the target labels by 0/1.
    """

    # Load the validation dataset
    validation_data = load_dataset(dataset_path)

    def normalize_validation(x_validation, x_min, x_max):
        """
        Normalize the features of the validation dataset.
        Use the minimum and the maximum values of each feature
        of the training set to normalize the validation set.
        All the features will be between 0 and 1.
        """
        x_validation_norm = (x_validation - x_min) / (x_max - x_min)
        return x_validation_norm.to_numpy()

    try:

        # Get the features and normalize them
        x_validation = validation_data.drop("Diagnosis", axis=1)
        x_validation_norm = normalize_validation(x_validation, x_min, x_max)

        # Get the target and replace the labels by 0/1
        y_validation = validation_data["Diagnosis"]
        y_validation = np.where(y_validation == "M", 0, 1)
        y_validation = y_validation.reshape(-1, 1)

        return x_validation_norm, y_validation

    except Exception:
        print("Error: can't get the validation data.")
        exit()


def create_layers_network(n_features,
                          layers,
                          activations,
                          learning_rate,
                          decay,
                          momentum):
    """
    Create a list of layers, used to init the MultiLayerPerceptron class
    Args:
    - n_features : number of dataset features (input for the Input Layer)
    - layers : list of number of input for each layer
    - activation : list of names of the activation function of each layer
    - learning_rate : initial learning rate
    - decay : decay used to minimize the learning rate during training
    - momentum : momentum of the layers, used to avoid local minima
    """

    try:

        layers_list = []
        n_layers = len(layers)
        for i in range(n_layers):
            n_input = n_features if i == 0 else layers[i - 1]
            layers_list.append(
                Dense_Layer(
                    n_inputs=n_input,
                    n_neurons=layers[i],
                    activation=activations[i],
                    learning_rate=learning_rate,
                    decay=decay,
                    momentum=momentum
                )
            )
            type = "Input" if i == 0 \
                else "Hidden" if i < n_layers - 1 \
                else "Output"
            print(f"{type} layer created.\n" +
                  f"Number of inputs: {n_input}\n" +
                  f"Number of neurons: {layers[i]}\n" +
                  f"Activation function: {activations[i]}\n" +
                  f"Learning rate: {learning_rate}\n")
        return layers_list

    except Exception:

        print("Error: cannot create the layer list.")
        exit()


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

    (
        layers,           # Number of outputs in each layer
        activations,      # Activation function in each layer
        loss_name,        # Loss function
        epochs,           # Number of epochs
        batch_size,       # Batch size
        learning_rate     # Learning rate
    ) = parse_args()

    # ##################################### #
    # 1- Load the datasets,                 #
    # 2- Get the features and the target,   #
    # 3- Normalize the features.            #
    # 4- Replace the target labels by 0/1,  #
    # ##################################### #

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

    n_features = x_train_norm.shape[1]
    n_train_samples = x_train_norm.shape[0]
    n_validation_samples = x_validation_norm.shape[0]

    print("Datasets loaded and normalized.\n\n" +
          f"Number of features: {n_features}\n" +
          f"Number of training samples: {n_train_samples}\n" +
          f"Number of validation samples: {n_validation_samples}\n")

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

    # Create the layers list
    layers_list = create_layers_network(
        n_features=n_features,
        layers=layers,
        activations=activations,
        learning_rate=learning_rate,
        decay=0.0005,   # Decay : learning_rate decreases over time
        momentum=0.05  # Momentum : avoid local minima and speed up SGD
    )

    multilayer_perceptron = MultiLayerPerceptron(
       layers=layers_list
    )

    loss_function = BinaryCrossEntropy_Loss()

    # ##################################### #
    # Train the neural network              #
    # ##################################### #

    n_batch = n_train_samples // batch_size
    if n_train_samples % batch_size != 0:
        n_batch += 1

    input("Press enter to continue...\n")

    metrics_functions = [
        accuracy_score_,
        precision_score_,
        recall_score_,
        f1_score_
    ]

    training_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
    }

    validation_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
    }

    learning_rates = []
    losses = []

    for epoch in ft_progress(range(epochs)):

        batch_losses = []

        batch_train_metrics = batch_validation_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
        }

        for i in range(n_batch):

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

        # Compute metrics on the validation set
        last_layer_output = multilayer_perceptron.forward(
            x_validation_norm
        )
        y_pred = np.argmax(last_layer_output, axis=1).reshape(-1, 1)
        for i, (metric, list_) in enumerate(validation_metrics.items()):
            list_.append(metrics_functions[i](y_validation, y_pred))

        # Append the batch metrics mean to training_metrics
        for i, (metric, list_) in enumerate(batch_train_metrics.items()):
            training_metrics[metric].append(np.mean(list_))

        losses.append(np.mean(batch_losses))
        learning_rates.append(
            multilayer_perceptron.layers[0].current_learning_rate
        )

        # TODO:
        # - Activation / Loss backward check
        # - Loss + Activation output in the same class ?

    # ############################### #
    # Final metrics on validation set #
    # ############################### #

    # Get the last element of the validation metrics lists and save them in a dict
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
    plot_loss(losses)

    # Accuracy evolution
    plot_metrics(
        training_metrics=training_metrics,
        validation_metrics=validation_metrics
    )

    # Plot the learning rate evolution
    plot_learning_rate(learning_rates)
