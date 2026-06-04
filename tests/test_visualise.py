import numpy as np
import pytest

from numcompute.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
)


def test_plot_metric_over_time_returns_figure_and_axis():
    fig, ax = plot_metric_over_time(
        [0.5, 0.6, 0.8],
        title="Accuracy Over Time",
        ylabel="Accuracy",
        show=False,
    )

    assert fig is not None
    assert ax is not None


def test_plot_metric_over_time_rejects_non_1d_input():
    with pytest.raises(ValueError):
        plot_metric_over_time(
            np.array([[0.5, 0.6]]),
            title="Bad",
            ylabel="Accuracy",
            show=False,
        )


def test_compare_models_returns_figure_and_axis():
    fig, ax = compare_models(
        [0.5, 0.6, 0.7],
        [0.4, 0.65, 0.75],
        labels=["Tree", "Ensemble"],
        show=False,
    )

    assert fig is not None
    assert ax is not None


def test_compare_models_shape_mismatch_raises_error():
    with pytest.raises(ValueError):
        compare_models(
            [0.5, 0.6],
            [0.5, 0.6, 0.7],
            labels=["Tree", "Ensemble"],
            show=False,
        )


def test_compare_models_invalid_labels_raises_error():
    with pytest.raises(ValueError):
        compare_models(
            [0.5, 0.6],
            [0.4, 0.7],
            labels=["Tree"],
            show=False,
        )


def test_plot_predictions_vs_ground_truth_returns_figure_and_axis():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    fig, ax = plot_predictions_vs_ground_truth(
        y_true,
        y_pred,
        show=False,
    )

    assert fig is not None
    assert ax is not None


def test_plot_predictions_vs_ground_truth_shape_mismatch():
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth(
            np.array([0, 1, 1]),
            np.array([0, 1]),
            show=False,
        )