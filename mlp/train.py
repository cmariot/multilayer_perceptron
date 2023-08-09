import pandas
import numpy as np
from layer import Layer_Dense
from losses import Categorical_Cross_Entropy
from multi_layer_perceptron import MultiLayerPerceptron
from metrics import Accuracy


def parse_args():
    pass


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


def train():
    pass


if __name__ == "__main__":

    # args = parse_args()

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

    # ##################################### #
    # Create the default neural network :   #
    # - Input layer                         #
    # - Hidden layer 1                      #
    # - Hidden layer 2                      #
    # - Output layer                        #
    # ##################################### #

    multilayer_perceptron = MultiLayerPerceptron(
        input_layer=Layer_Dense(
            n_inputs=x_train_norm.shape[1],
            n_neurons=30,
            activation="sigmoid"
        )
    )
    multilayer_perceptron.add_layer(
        n_neurons=24,
        activation="sigmoid"
    )
    multilayer_perceptron.add_layer(
        n_neurons=24,
        activation="sigmoid"
    )
    multilayer_perceptron.add_layer(
        n_neurons=2,
        activation="softmax"
    )

    # ##################################### #
    # Train the neural network              #
    # ##################################### #

    multilayer_perceptron.fit(
        x=x_train_norm,
        y=y_train,
        n_iter=1000,
        learning_rate=0.01,
        loss_=Categorical_Cross_Entropy(),
        accuracy_=Accuracy()
    )
