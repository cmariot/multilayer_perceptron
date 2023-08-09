import numpy as np
from activation_functions import activation_functions
from losses import Categorical_Cross_Entropy


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, activation):
        # We need to initialize the weights and the biais.
        # The weights are a matrix of shape (n_inputs, n_neurons).
        # The biais is a vector of shape (n_neurons, 1).
        # The weights and the biais are randomly initialized.
        # The weights are multiplied by 0.1 to avoid large values.
        # We will not need to transpose the weights matrix.
        # Weights and biais are the parameters of the model.
        # They are updated during the training.
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biais = np.zeros((1, n_neurons))
        self.activation = activation_functions[activation]

    def forward(self, inputs):
        # Inputs are the outputs of the previous layer.
        # The weighted sum is a matrix of shape (n_samples, n_neurons).
        # The output is a matrix of shape (n_samples, n_neurons).
        self.weighted_sum = np.dot(inputs, self.weights) + self.biais
        self.output = self.activation(self.weighted_sum)
        return self.output


if __name__ == "__main__":

    # X is a matrix of shape (3, 4).
    # 3 samples, 4 features.
    X = np.array([
        [1, 2, 3, 2.5],
        [2, 5, -1, 2],
        [-1.5, 2.7, 3.3, -0.8]
    ])

    y_true = np.array([
        [0, 1],
        [1, 0],
        [0, 1]
    ])

    # The number of samples is the number of rows.
    # The number of inputs is the number of features,
    # or the number of output of the previous layer.
    n_samples, n_inputs = X.shape

    # We want 24 neurons for the first layer.
    n_neurons = 24
    activation_function = "sigmoid"
    layer1 = Layer_Dense(n_inputs, n_neurons, activation_function)
    layer1.forward(X)
    # print(layer1.output)

    # The number of inputs is the number of outputs of the previous layer.
    n_inputs = layer1.output.shape[1]
    n_neurons = 24
    activation_function = "sigmoid"
    layer2 = Layer_Dense(n_inputs, n_neurons, activation_function)
    layer2.forward(layer1.output)
    # print(layer2.output)

    # The number of inputs is the number of outputs of the previous layer.
    n_inputs = layer2.output.shape[1]
    n_neurons = 2
    activation_function = "softmax"
    layer3 = Layer_Dense(n_inputs, n_neurons, activation_function)
    layer3.forward(layer2.output)
    print(layer3.output)

    loss_function = Categorical_Cross_Entropy()
    loss = loss_function.calculate(y_true, layer3.output)
    print(loss)
