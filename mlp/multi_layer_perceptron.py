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
            layer.activation.forward(layer.weighted_sum)
            input = layer.activation.output
        return self.layers[-1].activation.output

    def backward(self, output_error):
        """
        Backward propagation.
        """
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error)
        return output_error

    def update_parameters(self):
        """
        Update parameters.
        """
        for layer in self.layers:
            layer.update()
