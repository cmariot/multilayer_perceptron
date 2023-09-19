import numpy as np
import nnfs
from Activation.softmax_categorical_cross_entropy import Softmax_Categorical_Cross_Entropy
from Activation.softmax import SoftmaxActivation
from Activation.categorical_cross_entropy import CategoricalCrossEntropy_Loss


if __name__ == "__main__":

    # # #################### #
    # # Network architecture #
    # # #################### #

    # # Input to the network
    # x = np.array([1.0, -2.0, 3.0])

    # # Weights and biases
    # weights = np.array([-3.0, -1.0, 2.0])
    # biases = 1.0

    # # ############ #
    # # Forward pass #
    # # ############ #

    # # Multiplying inputs by weights
    # ws0 = x[0] * weights[0]
    # ws1 = x[1] * weights[1]
    # ws2 = x[2] * weights[2]
    # print("ws0: ", ws0)
    # print("ws1: ", ws1)
    # print("ws2: ", ws2)

    # # Summing weighted inputs and adding bias
    # z = ws0 + ws1 + ws2 + biases
    # print("z: ", z)

    # # ReLU activation function
    # a = np.maximum(z, 0)

    # # ############# #
    # # Backward pass #
    # # ############# #

    # # Derivative from the next layer
    # dvalue = 1.

    # # Derivative of Relu and the chain rule
    # drelu_dz = dvalue * (1 if a > 0 else 0)

    # # Partial derivatives of the multiplication, the chain rule
    # dsum_dxw0 = 1
    # dsum_dxw0 = drelu_dz * dsum_dxw0


    # ###################################################################
    # Comparaison between the Softmax Categorical Cross-Entropy Class and
    # the Softmax + Categorical Cross-Entropy in two different Classes
    # ###################################################################

    nnfs.init()

    softmax_output = np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.5, 0.4],
        [0.02, 0.9, 0.08]
    ])

    class_targets = np.array([0, 1, 1])

    softmax_loss = Softmax_Categorical_Cross_Entropy()
    softmax_loss.backward(softmax_output, class_targets)
    dvalues1 = softmax_loss.dinputs
    print(dvalues1)

    activation = SoftmaxActivation()
    activation.output = softmax_output
    loss = CategoricalCrossEntropy_Loss()
    loss.backward(softmax_output, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs

    print(dvalues2)
