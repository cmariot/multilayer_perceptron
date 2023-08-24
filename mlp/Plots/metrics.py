import matplotlib.pyplot as plt


def plot_metrics(training_metrics: dict, validation_metrics: dict):
    """
    Plot the metrics evolution.
    """

    try:

        colors = ["b", "g", "r", "y"]

        for i, (metric_name, metric_list) in enumerate(training_metrics.items()):
            plt.plot(metric_list, label=f"{metric_name} training", color=colors[i])
        for i, (metric_name, metric_list) in enumerate(validation_metrics.items()):
            plt.plot(metric_list, label=f"{metric_name} validation", color=colors[i], linestyle=":")
        plt.title("Metrics evolution")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(-0.05, 1.05)
        plt.grid()
        plt.legend()
        plt.show()

    except Exception as e:

        print(f"Error while plotting the loss: {e}")
        exit()
