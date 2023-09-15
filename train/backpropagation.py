import numpy as np


if __name__ == "__main__":

    # #################### #
    # Network architecture #
    # #################### #

    # Input to the network
    x = np.array([1.0, -2.0, 3.0])

    # Weights and biases
    weights = np.array([-3.0, -1.0, 2.0])
    biases = 1.0

    # ############ #
    # Forward pass #
    # ############ #

    # Multiplying inputs by weights
    ws0 = x[0] * weights[0]
    ws1 = x[1] * weights[1]
    ws2 = x[2] * weights[2]
    print("ws0: ", ws0)
    print("ws1: ", ws1)
    print("ws2: ", ws2)

    # Summing weighted inputs and adding bias
    z = ws0 + ws1 + ws2 + biases
    print("z: ", z)

    # ReLU activation function
    a = np.maximum(z, 0)

    # ############# #
    # Backward pass #
    # ############# #

    # Derivative from the next layer
    dvalue = 1.

    # Derivative of Relu and the chain rule
    drelu_dz = dvalue * (1 if a > 0 else 0)

    # Partial derivatives of the multiplication, the chain rule
    dsum_dxw0 = 1
    dsum_dxw0 = drelu_dz * dsum_dxw0
