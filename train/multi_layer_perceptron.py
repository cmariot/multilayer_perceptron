# *************************************************************************** #
#                                                                             #
#                                                       :::      ::::::::     #
#   multi_layer_perceptron.py                         :+:      :+:    :+:     #
#                                                   +:+ +:+         +:+       #
#   By: cmariot <cmariot@student.42.fr>           +#+  +:+       +#+          #
#                                               +#+#+#+#+#+   +#+             #
#   Created: 2023/08/24 14:39:36 by cmariot          #+#    #+#               #
#   Updated: 2023/08/24 14:39:37 by cmariot         ###   ########.fr         #
#                                                                             #
# *************************************************************************** #


from layer import Dense_Layer
import numpy as np
from Loss.binary_cross_entropy import BinaryCrossEntropy_Loss
import pandas


losses = {
    "binaryCrossentropy": BinaryCrossEntropy_Loss
}


class MultiLayerPerceptron:

    def __init__(
            self,
            n_features,
            n_neurons,
            activations,
            learning_rate,
            decay,
            momentum,
            batch_size,
            n_train_samples,
            loss_name):

        try:

            # Create the layer list
            self.layers = []
            n_layers = len(n_neurons)
            for i in range(n_layers):
                n_input = n_features if i == 0 else n_neurons[i - 1]
                self.layers.append(
                    Dense_Layer(
                        n_inputs=n_input,
                        n_neurons=n_neurons[i],
                        activation=activations[i],
                        learning_rate=learning_rate,
                        decay=decay,
                        momentum=momentum
                    )
                )

            # Create the loss function
            if loss_name in losses:
                self.loss_function = losses[loss_name]()
            else:
                print("Error: unknown loss function.")
                exit()

            # Compute the number of batches
            self.n_batch = n_train_samples // batch_size
            if n_train_samples % batch_size != 0:
                self.n_batch += 1

            # Create a list to save the learning rates
            self.learning_rates = []

            # Print the layer list in a dataframe
            df = {
                "Number of inputs": [n_features] + n_neurons[:-1],
                "Number of neurons": n_neurons,
                "Weight initialization": ["heUniform"] * n_layers,
                "Bias initialization": ["ones"] * n_layers,
                "Activation function": activations,
                "Learning rate": [learning_rate] * n_layers,
                "Decay": [decay] * n_layers,
                "Momentum": [momentum] * n_layers
            }
            index = ["Input Layer"] + \
                [f"Hidden Layer {i + 1}" for i in range(n_layers - 2)] \
                + ["Output Layer"]
            df = pandas.DataFrame(df, index=index).transpose()
            print(df, "\n")

        except Exception:
            print("Error: cannot create the layer list.")
            exit()

    def forward(self, input):
        """
        Forward propagation.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def predict(self, output):
        """
        Predict.
        """
        y_hat = np.argmax(output, axis=1).reshape(-1, 1)
        return y_hat

    def loss(self, y_hat, y):
        """
        Loss.
        """
        return self.loss_function.forward(y_hat, y)

    def gradient(self, output, y):
        """
        Gradient.
        """
        return self.loss_function.gradient(output, y)

    def backward(self, dinput):
        """
        Backward propagation.
        """
        for layer in reversed(self.layers):
            dinput = layer.backward(dinput)
        return dinput

    def update_learning_rate(self):
        """
        Update learning rate.
        """
        for layer in self.layers:
            layer.update_learning_rate()
            layer.update_iterations()
        self.learning_rates.append(
            self.layers[0].current_learning_rate
        )

    def gradient_descent(self):
        """
        Update parameters.
        """
        for layer in self.layers:
            layer.gradient_descent()
