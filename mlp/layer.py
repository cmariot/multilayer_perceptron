import numpy as np


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # We need to initialize the weights and the biais.
        # The weights are a matrix of shape (n_inputs, n_neurons).
        # The biais is a vector of shape (n_neurons, 1).
        # The weights and the biais are randomly initialized.
        # The weights are multiplied by 0.1 to avoid large values.
        # We will not need to transpose the weights matrix.
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biais = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biais


if __name__ == "__main__":

    # X is a matrix of shape (3, 4).
    # 3 samples, 4 features.
    X = np.array([
        [1, 2, 3, 2.5],
        [2, 5, -1, 2],
        [-1.5, 2.7, 3.3, -0.8]
    ])

    # The number of samples is the number of rows.
    # The number of inputs is the number of features,
    # or the number of output of the previous layer.
    n_samples, n_inputs = X.shape

    # We want 5 neurons for the first layer.
    n_neurons = 5
    layer1 = Layer_Dense(n_inputs, n_neurons)
    layer1.forward(X)
    # print(layer1.output)

    # The number of inputs is the number of outputs of the previous layer.
    n_inputs = layer1.output.shape[1]
    layer2 = Layer_Dense(n_inputs, n_neurons)
    layer2.forward(layer1.output)
    print(layer2.output)
