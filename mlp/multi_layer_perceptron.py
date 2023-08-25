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
                type = "Input" if i == 0 \
                    else "Hidden" if i < n_layers - 1 \
                    else "Output"
                print(f"{type} layer created.\n" +
                      f"Number of inputs: {n_input}\n" +
                      f"Number of neurons: {n_neurons[i]}\n" +
                      f"Activation function: {activations[i]}\n" +
                      f"Learning rate: {learning_rate}\n")

            if loss_name in losses:
                self.loss_function = losses[loss_name]()
            else:
                print("Error: unknown loss function.")
                exit()

            self.n_batch = n_train_samples // batch_size
            if n_train_samples % batch_size != 0:
                self.n_batch += 1

            self.learning_rates = [self.layers[0].learning_rate]

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
