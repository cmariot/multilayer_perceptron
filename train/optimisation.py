import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
from dense_layer import DenseLayer
from Activation.relu import ReluActivation
from Activation.softmax import SoftmaxActivation
from Loss.categorical_cross_entropy import CategoricalCrossEntropy_Loss


if __name__ == "__main__":

    nnfs.init()

    X, y = vertical_data(samples=100, classes=3)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    # Layers
    layer1 = DenseLayer(2, 3)
    activation1 = ReluActivation()
    layer2 = DenseLayer(3, 3)
    activation2 = SoftmaxActivation()

    loss = CategoricalCrossEntropy_Loss()

    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    current_loss = loss.calculate(activation2.output, y)

    print(current_loss)
