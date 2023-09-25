from layer import Layer
from Loss.binary_cross_entropy import BinaryCrossEntropy_Loss
from Optimizers.sgd import StandardGradientDescent
import pickle


class MultilayerPerceptron:

    available_losses = {
        "binaryCrossentropy": BinaryCrossEntropy_Loss(),
    }

    available_optimizers = {
        "sgd": StandardGradientDescent,
    }

    def __init__(self,
                 n_features,
                 n_neurons,
                 activations,
                 loss_name,
                 epochs,
                 batch_size,
                 x_min,
                 x_max,
                 n_train_samples,
                 learning_rate,
                 decay,
                 momentum):

        try:

            # Layers initialization
            self.layers = []
            for i in range(len(n_neurons)):

                n_input = n_features if i == 0 else n_neurons[i - 1]
                n_output = n_neurons[i]
                activation = activations[i]

                self.layers.append(
                    Layer(
                        n_input,
                        n_output,
                        activation
                    )
                )

            # Loss function initialization
            if loss_name not in self.available_losses:
                raise Exception("Loss function not available")
            self.loss = self.available_losses[loss_name]

            # Optimizer initialization
            self.optimizer = StandardGradientDescent(
                learning_rate,
                decay,
                momentum
            )

            # Learning rates evolution
            self.learning_rates = []

            # Training parameters initialization
            self.epochs = epochs
            self.batch_size = batch_size
            self.n_batch = n_train_samples // batch_size
            if n_train_samples % batch_size != 0:
                self.n_batch += 1

            # Normalization parameters
            self.x_min = x_min
            self.x_max = x_max

        except Exception as e:
            print(e)
            exit()


    def forward(self, inputs):
        try:
            for layer in self.layers:
                weigthed_sum = layer.forward(inputs)
                output = layer.activation_function.forward(weigthed_sum)
                inputs = output
            return output
        except Exception as e:
            print(e)
            exit()

    def backward(self, y):
        try:
            dinputs = self.loss.backward(self.layers[-1].activation_function.output, y)
            for i, layer in enumerate(reversed(self.layers)):
                layer.activation_function.backward(dinputs)
                layer.backward(layer.activation_function.dinputs)
                dinputs = layer.dinputs

        except Exception as e:
            print(e)
            exit()

    def optimize(self):
        self.optimizer.pre_update_params()
        self.learning_rates.append(self.optimizer.current_learning_rate)
        for layer in self.layers:
            self.optimizer.update(layer)
        self.optimizer.post_update_params()

    def save_model(self, path):
        try:
            with open(path, "wb") as file:
                pickle.dump(self, file)
        except Exception as e:
            print(e)
            exit()

    def load_model(self, path):
        try:
            with open(path, "rb") as file:
                self = pickle.load(file)
        except Exception as e:
            print(e)
            exit()