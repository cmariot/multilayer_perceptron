from layer import Layer
import numpy as np


class Model:

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == "__main__":

    x = np.array([
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ])

    layer1 = Layer(
        n_inputs=4,
        n_neurons=5,
        activation_function="relu"
    )

    layer2 = Layer(
        n_inputs=5,
        n_neurons=2,
        activation_function="softmax"
    )

    model = Model([layer1, layer2])

    output = model.forward(x)

    print(output) 