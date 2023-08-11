import argparse
import pandas
import numpy as np
from losses import loss_selection
from multi_layer_perceptron import MultiLayerPerceptron
from layer import Dense_Layer


def parse_args():
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--layers",
            type=int,
            nargs="+",
            help="Number of neurons in each layer",
            default=[30, 24, 24, 2]
        )

        parser.add_argument(
            "--activations",
            type=str,
            nargs="+",
            help="Activation function in each layer",
            default=["sigmoid", "sigmoid", "sigmoid", "softmax"]
        )

        parser.add_argument(
            "--epochs",
            type=int,
            help="Number of epochs",
            default=84
        )

        parser.add_argument(
            "--loss",
            type=str,
            help="Loss function",
            default="binaryCrossentropy"
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            help="Batch size",
            default=8
        )

        parser.add_argument(
            "--learning_rate",
            type=float,
            help="Learning rate",
            default=10e-3
        )

        args = parser.parse_args()

        return (
            args.layers,
            args.activations,
            args.epochs,
            args.loss,
            args.batch_size,
            args.learning_rate
        )

    except Exception as e:
        print(e)
        exit()


def load_dataset(path):
    try:
        dataset = pandas.read_csv(path)
        return dataset
    except Exception as e:
        print(e)
        exit()


def normalize_train(x_train):
    x_min = x_train.min()
    x_max = x_train.max()
    x_train_norm = (x_train - x_min) / (x_max - x_min)
    return x_train_norm.to_numpy(), x_min, x_max


def normalize_validation(x_validation, x_min, x_max):
    x_validation_norm = (x_validation - x_min) / (x_max - x_min)
    return x_validation_norm.to_numpy()


def categorical_target_to_numerical(y_train, y_validation):
    """
    Encode categorical target to numerical values
    0: Benign
    1: Malignant
    """
    y_train = np.where(y_train == "M", 1, 0)
    y_validation = np.where(y_validation == "M", 1, 0)
    return (
        y_train.reshape(-1, 1),
        y_validation.reshape(-1, 1)
    )


def create_layers_network(n_features,
                          layers,
                          activations,
                          learning_rate):
    layers_list = []
    n_layers = len(layers)
    for i in range(n_layers):
        if i == 0:
            # Create the input layer
            layers_list.append(
                Dense_Layer(
                    n_inputs=n_features,
                    n_neurons=layers[i],
                    activation=activations[i],
                    learning_rate=learning_rate
                )
            )

        else:
            # Create the hidden / output layers
            layers_list.append(
                Dense_Layer(
                    n_inputs=layers[i - 1],
                    n_neurons=layers[i],
                    activation=activations[i],
                    learning_rate=learning_rate
                )
            )

        type = "Input" if i == 0 \
            else "Hidden" if i < n_layers - 1 \
            else "Output"

        print(f"{type} layer created.\n" +
              f"Number of neurons: {layers[i]}\n" +
              f"Activation function: {activations[i]}\n" +
              f"Learning rate: {learning_rate}\n")
    return layers_list


def get_batch(x, y, i, batch_size):
    start = i * batch_size
    end = (i + 1) * batch_size
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def train():
    pass


if __name__ == "__main__":

    (
        layers,
        activations,
        epochs,
        loss_name,
        batch_size,
        learning_rate
    ) = parse_args()

    # ##################################### #
    # 1- Load the datasets,                 #
    # 2- Get the features and the target,   #
    # 3- Normalize the features.            #
    # 4- Replace the target labels by 0/1,  #
    # ##################################### #

    train_data = load_dataset("../datasets/train.csv")
    validation_data = load_dataset("../datasets/validation.csv")

    x_train = train_data.drop("Diagnosis", axis=1)
    y_train = train_data["Diagnosis"]

    x_validation = validation_data.drop("Diagnosis", axis=1)
    y_validation = validation_data["Diagnosis"]

    x_train_norm, x_min, x_max = normalize_train(x_train)
    x_validation_norm = normalize_validation(x_validation, x_min, x_max)

    y_train, y_validation = categorical_target_to_numerical(
        y_train, y_validation
    )

    print("Datasets loaded and normalized.\n" +
          f"Number of features: {x_train_norm.shape[1]}\n" +
          f"Number of training samples: {x_train_norm.shape[0]}\n" +
          f"Number of validation samples: {x_validation_norm.shape[0]}\n")

    # ############################# #
    # Create the neural network :   #
    # - Input layer                 #
    # - Hidden layer 1              #
    # - Hidden layer 2              #
    # - Output layer                #
    # ############################# #

    # Create the layers list
    layers_list = create_layers_network(
        n_features=x_train_norm.shape[1],
        layers=layers,
        activations=activations,
        learning_rate=learning_rate
    )

    multilayer_perceptron = MultiLayerPerceptron(
       layers=layers_list
    )

    loss = loss_selection(loss_name)

    # ##################################### #
    # Train the neural network              #
    # ##################################### #

    for epoch in range(epochs):

        n_batch = x_train_norm.shape[0] // batch_size

        for i in range(n_batch):

            x_batch, y_batch = get_batch(
                x_train_norm, y_train, i, batch_size
            )

            # Forward pass
            last_layer_output = multilayer_perceptron.forward(x_batch)

            # # Calculate the loss
            # data_loss = loss.calculate(last_layer_output, y_batch)

            # # Calculate the accuracy
            # predictions = np.argmax(last_layer_output, axis=1)
            # accuracy = np.mean(predictions == y_batch)

            # print(f"epoch: {epoch}, batch: {i}, " +
            #       f"acc: {accuracy:.3f}, loss: {data_loss:.3f}")

            # # Backward pass
            # loss.backward(multilayer_perceptron.output, y_batch)
            # multilayer_perceptron.backward(loss.dinputs)

            # # Update the weights and the biases
            # multilayer_perceptron.update_parameters()

        # TODO: Compute metrics on the validation set
