import pandas
import numpy as np
from parse_arguments import parse_args
from Loss.binary_cross_entropy import BinaryCrossEntropy_Loss
from multi_layer_perceptron import MultiLayerPerceptron
from layer import Dense_Layer
from matplotlib import pyplot as plt


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

    # TODO : Features selection ?

    x_train = train_data.drop("Diagnosis", axis=1)
    # Select the 10 last features
    x_train = x_train.iloc[:, -10:]
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
    # - Input layer : sigmoid       #
    # - Hidden layer 1 : sigmoid    #
    # - Hidden layer 2 : sigmoid    #
    # - Output layer : softmax      #
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

    loss_function = BinaryCrossEntropy_Loss()

    # ##################################### #
    # Train the neural network              #
    # ##################################### #

    losses = []
    accuracies = []

    n_batch = x_train_norm.shape[0] // batch_size

    for epoch in range(epochs):

        batch_losses = []
        batch_accuracies = []

        for i in range(n_batch):

            x_batch, y_batch = get_batch(
                x_train_norm, y_train, i, batch_size
            )

            # Forward pass
            last_layer_output = multilayer_perceptron.forward(x_batch)

            # Compute the loss
            loss = loss_function.compute(last_layer_output, y_batch)

            # Get predictions
            y_pred = np.argmax(last_layer_output, axis=1)

            # Compute the accuracy
            accuracy = np.mean(y_pred == y_batch)

            print(f"epoch: {epoch}, " +
                  f"acc: {accuracy:.3f}, loss: {loss:.5f}")

            # Save the current loss and accuracy, used for the plot
            batch_losses.append(loss)
            batch_accuracies.append(accuracy)

            # calculating the derivative of cost with respect to some weight
            dcost = loss_function.gradient(last_layer_output, y_batch)

            # Backpropagation
            for layer in reversed(multilayer_perceptron.layers):
                dcost = layer.backward(dcost)

            # Update the weights and the biases
            multilayer_perceptron.update_parameters()

        losses.append(np.mean(batch_losses))
        accuracies.append(np.mean(batch_accuracies))

        # TODO:
        # - Understand how to use the gradient (loss backward)
        # - Backpropagation : Activation backward +
        #                     Layer backward for each layer ?
        # - Gradient descent
        # - Batch
        # - Compute metrics on the validation set

    # ##################################### #
    # Plot the loss and the accuracy        #
    # ##################################### #

    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    plt.plot(accuracies)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid()
    plt.show()
