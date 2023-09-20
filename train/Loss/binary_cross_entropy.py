from Loss.loss import Loss
import numpy as np

class BinaryCrossEntropy_Loss(Loss):

    def forward(self, output, y):
        output_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        sample_losses = -(y * np.log(output_clipped) + (1 - y) * np.log(1 - output_clipped))
        self.output = np.mean(sample_losses, axis=-1)
        return self.output

    def backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y / dvalues_clipped - (1 - y) / (1 - dvalues_clipped)) / labels
        self.dinputs = self.dinputs / samples
        return self.dinputs


if __name__ == "__main__":

    softmax_output = np.array([[0.7, 0.1, 0.2],
                               [0.1, 0.5, 0.4],
                               [0.02, 0.9, 0.08]])

    class_targets = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 1, 0]])

    loss = BinaryCrossEntropy_Loss()

    print(loss.calculate(softmax_output, class_targets))
