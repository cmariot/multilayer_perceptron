import matplotlib.pyplot as plt


def plot_loss(losses):
    """
    Plot the loss evolution.
    """

    try:

        plt.plot(losses, label="Training")
        plt.title("Loss evolution")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.show()

    except Exception as e:

        print(f"Error while plotting the loss: {e}")
        exit()
