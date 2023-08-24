import matplotlib.pyplot as plt


def plot_metrics(training_metrics: dict):
    """
    Plot the metrics evolution.
    """

    try:

        for metric_name, metric_list in training_metrics.items():
            plt.plot(metric_list, label=f"{metric_name} training")
            if metric_name == "f1_score":
                print(f"Max f1_score: {max(metric_list)}")
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
