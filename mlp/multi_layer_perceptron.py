from layer import Layer_Dense


class MultiLayerPerceptron:

    def __init__(self, input_layer):
        self.layers = [input_layer]

    def add_layer(self, n_neurons, activation):
        """
        Add a layer to the network.
        The output of the previous layer is the input of the new layer.
        """
        try:
            n_inputs = self.layers[-1].weights.shape[1]
            self.layers.append(
                Layer_Dense(
                    n_inputs=n_inputs,
                    n_neurons=n_neurons,
                    activation=activation
                )
            )
        except Exception as e:
            print("Add Layer Exception :", e)
            exit(1)

    def forward(self, x):
        """
        Forward propagation.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self, x, y, n_iter, learning_rate, loss_, accuracy_):

        for i in range(n_iter):

            y_hat = self.forward(x)
            loss = loss_.calculate(y, y_hat)
            accu = accuracy_.calculate(y, y_hat)

            print(f"iter : {i} | loss : {loss} | accuracy : {accu}")

            # Backward propagation
            # Update weights and biases
