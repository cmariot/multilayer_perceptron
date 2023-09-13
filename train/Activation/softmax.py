import numpy as np


class SoftmaxActivation:
    """
    Softmax activation function is usually used
    in the output layer of a classification model
    It returns a value between 0 and 1,
    1 being the highest probability
    """

    def forward(self, layer_output):
        exp_values = np.exp(
            layer_output -
            np.max(layer_output, axis=1, keepdims=True)
        )
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output


if __name__ == "__main__":

    import nnfs
    from nnfs.datasets import spiral_data
    from dense_layer import DenseLayer
    from relu import ReluActivation

    nnfs.init()

    X, y = spiral_data(samples=100, classes=3)

    dense1 = DenseLayer(n_inputs=2, n_neurons=3)
    activation1 = ReluActivation()

    dense2 = DenseLayer(n_inputs=3, n_neurons=3)
    activation2 = SoftmaxActivation()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(activation2.output[:5])
