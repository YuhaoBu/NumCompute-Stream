import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(metric_values, title, ylabel, save_path=None, show=True):
    values = np.asarray(metric_values, dtype=float)

    if values.ndim != 1:
        raise ValueError("metric_values must be a 1D array or list.")

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(values) + 1), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def compare_models(metric1, metric2, labels, title="Model Comparison", ylabel="Metric", save_path=None, show=True):
    metric1 = np.asarray(metric1, dtype=float)
    metric2 = np.asarray(metric2, dtype=float)

    if metric1.ndim != 1 or metric2.ndim != 1:
        raise ValueError("metric1 and metric2 must be 1D arrays or lists.")

    if metric1.shape != metric2.shape:
        raise ValueError("metric1 and metric2 must have the same shape.")

    if len(labels) != 2:
        raise ValueError("labels must contain exactly two model names.")

    x = np.arange(1, len(metric1) + 1)

    fig, ax = plt.subplots()
    ax.plot(x, metric1, marker="o", label=labels[0])
    ax.plot(x, metric2, marker="s", label=labels[1])
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs Ground Truth", save_path=None, show=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")

    x = np.arange(len(y_true))

    fig, ax = plt.subplots()
    ax.plot(x, y_true, marker="o", linestyle="-", label="Ground Truth")
    ax.plot(x, y_pred, marker="x", linestyle="--", label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class")
    ax.legend()
    ax.grid(True)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax