class MultiLayerPerceptron:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        """
        Forward propagation.
        """
        for layer in self.layers:
            # Compute weighted sum
            layer.forward(input)
            # Compute activation function output
            layer.activation_forward()
            input = layer.output
        return self.layers[-1].output

    def backward(self, gradient, y_true):
        """
        Backward propagation.
        """
        for layer in reversed(self.layers):
            layer.backward(gradient)

    def update_parameters(self):
        """
        Update parameters.
        """
        for layer in self.layers:
            layer.update()
