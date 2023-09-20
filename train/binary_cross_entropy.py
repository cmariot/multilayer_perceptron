from Loss.loss import Loss
import numpy as np


class BinaryCrossEntropy_Loss(Loss):

    # Compute the model error
    def forward(self, output, y):
        output_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        sample_losses = -(y * np.log(output_clipped) + (1 - y) * np.log(1 - output_clipped))
        self.output = np.mean(sample_losses, axis=-1)
        return self.output

    # Used during the training phase
    def backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y / dvalues_clipped - (1 - y) / (1 - dvalues_clipped)) / labels
        self.dinputs = self.dinputs / samples
        return self.dinputs


if __name__ == "__main__":

    # Last layer output (y_hat)
    softmax_output = np.array(
            [
                [0.70, 0.30],
                [0.10, 0.90],
                [0.02, 0.98]
            ]
    )

    # Real targets (y)
    class_targets = np.array(
            [
                [1, 0],
                [0, 1],
                [0, 1]
            ]
    )

    # Instance of the loss class
    loss_class = BinaryCrossEntropy_Loss()

    # Loss (cost)
    loss = loss.calculate(softmax_output, class_targets)

    print(loss)

