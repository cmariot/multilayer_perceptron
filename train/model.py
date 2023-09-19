from layer import Layer
import numpy as np
from Loss.categorical_cross_entropy import CategoricalCrossEntropy_Loss


class Model:

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dvalue, y):
        # dvalue est l'output de Loss backward sur l'output du dernier layer
        for layer in reversed(self.layers):
            dvalue = layer.backward(dvalue)
        return dvalue

    def update(self):
        for layer in self.layers:
            layer.update()


if __name__ == "__main__":

    x = np.array([
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ])

    y = np.array([
        [1, 1, 1],
        [0, 2, 1],
        [1, 2, 1]
    ])

    layer1 = Layer(
        n_inputs=4,
        n_neurons=5,
        activation_function="relu"
    )

    layer2 = Layer(
        n_inputs=5,
        n_neurons=3,
        activation_function="softmax"
    )

    model = Model([layer1, layer2])

    output = model.forward(x)

    print(output)

    loss_function = CategoricalCrossEntropy_Loss()
    loss = loss_function.calculate(output, y)
    print(loss)

    dvalue = loss_function.backward(output, y)
    model.backward(dvalue, y)

    model.update()

    output = model.forward(x)
    loss = loss_function.calculate(output, y)
    print(loss)

