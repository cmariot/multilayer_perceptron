from Layer import Layer


class MultilayerPerceptron:

    def __init__(self, n_neurons):

        self.layers = []
        for i in range(len(n_neurons)):
            layer = Layer(n_neurons[i])
            self.layers.append(layer)
