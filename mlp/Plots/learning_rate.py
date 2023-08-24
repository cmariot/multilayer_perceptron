import matplotlib.pyplot as plt


def plot_learning_rate(learning_rates):
    """
    Plot the learning rate evolution.
    """

    try:

        plt.plot(learning_rates)
        plt.title("Learning rate evolution")
        plt.ylabel("Learning rate")
        plt.xlabel("Epochs")
        plt.grid()
        plt.show()

    except Exception as e:

        print(f"Error while plotting the learning rate: {e}")
        exit()
