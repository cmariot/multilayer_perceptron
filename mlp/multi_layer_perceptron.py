# ****************************************************************************#
#                                                                             #
#                                                        :::      ::::::::    #
#   multi_layer_perceptron.py                          :+:      :+:    :+:    #
#                                                    +:+ +:+         +:+      #
#   By: cmariot <cmariot@student.42.fr>            +#+  +:+       +#+         #
#                                                +#+#+#+#+#+   +#+            #
#   Created: 2023/08/24 14:39:36 by cmariot           #+#    #+#              #
#   Updated: 2023/08/24 14:39:37 by cmariot          ###   ########.fr        #
#                                                                             #
# ****************************************************************************#


from layer import Dense_Layer


class MultiLayerPerceptron:

    def __init__(self,
                 n_features,
                 layers,
                 activations,
                 learning_rate,
                 decay,
                 momentum,
                 batch_size,
                 n_train_samples
                 ):

        try:
            self.layers = []
            n_layers = len(layers)
            for i in range(n_layers):
                n_input = n_features if i == 0 else layers[i - 1]
                self.layers.append(
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

            self.n_batch = n_train_samples // batch_size
            if n_train_samples % batch_size != 0:
                self.n_batch += 1

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

    def update_parameters(self):
        """
        Update parameters.
        """
        for layer in self.layers:
            layer.update()

    def update_iterations(self):
        """
        Update iterations.
        """
        for layer in self.layers:
            layer.update_iterations()
