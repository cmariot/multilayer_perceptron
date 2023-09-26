import numpy as np


class BinaryCrossEntropy_Loss:

    # Compute the model error
    def calculate(self, output, y):
        output_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        sample_losses = -(y * np.log(output_clipped) + (1 - y) * np.log(1 - output_clipped))
        self.output = np.mean(sample_losses, axis=-1)
        loss = np.mean(self.output)
        return loss

    # Used during the training phase
    def backward(self, dvalues, y):
        samples = len(dvalues)
        labels = len(dvalues[0])
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y / dvalues_clipped - (1 - y) / (1 - dvalues_clipped)) / labels
        self.dinputs = self.dinputs / samples
        return self.dinputs

