import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    precision,
    recall,
    f1,               # 如果你实现的是 f1_score，请改成 f1_score
    confusion_matrix,
    mse,
)


def test_accuracy_basic():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    assert accuracy(y_true, y_pred) == 0.75


def test_precision_basic():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    assert np.isclose(precision(y_true, y_pred), 1.0)


def test_recall_basic():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    assert np.isclose(recall(y_true, y_pred), 2 / 3)


def test_f1_basic():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    p = 1.0
    r = 2 / 3
    expected = 2 * p * r / (p + r)
    assert np.isclose(f1(y_true, y_pred), expected)


def test_confusion_matrix_binary():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    cm = confusion_matrix(y_true, y_pred)
    # [[TN, FP],
    #  [FN, TP]]
    expected = np.array([[1, 1],
                         [1, 1]])
    assert cm.shape == (2, 2)
    assert np.array_equal(cm, expected)


def test_mse_basic():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])
    assert np.isclose(mse(y_true, y_pred), 1 / 3)


def test_all_correct():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0, 1])
    assert accuracy(y_true, y_pred) == 1.0
    assert precision(y_true, y_pred) == 1.0
    assert recall(y_true, y_pred) == 1.0
    assert f1(y_true, y_pred) == 1.0


def test_no_positive_predictions_precision_zero():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert precision(y_true, y_pred) == 0.0


def test_no_positive_labels_recall_zero():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 0, 1])
    assert recall(y_true, y_pred) == 0.0


def test_shape_mismatch_classification():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0])
    with pytest.raises(ValueError):
        accuracy(y_true, y_pred)


def test_shape_mismatch_mse():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])
    with pytest.raises(ValueError):
        mse(y_true, y_pred)

# =====================================
# Streaming Metrics Tests
# =====================================

from numcompute.metrics import StreamingClassificationMetrics

def test_streaming_metrics_single_chunk():
    metric = StreamingClassificationMetrics()

    metric.update(
        [1, 1, 0, 1],
        [1, 0, 0, 1]
    )

    assert metric.accuracy() == 0.75


def test_streaming_metrics_multiple_chunks():
    metric = StreamingClassificationMetrics()

    metric.update([1, 1], [1, 0])
    metric.update([0, 1], [0, 1])

    assert metric.accuracy() == 0.75


def test_streaming_metrics_reset():
    metric = StreamingClassificationMetrics()

    metric.update([1, 1], [1, 0])

    metric.reset()

    assert metric.accuracy() == 0.0

def test_streaming_auc_accumulates_with_scores():
    metric = StreamingClassificationMetrics()

    metric.update(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
        y_score=np.array([0.1, 0.8, 0.3, 0.9])
    )

    assert np.isclose(metric.auc(), 1.0)


def test_streaming_auc_uses_predictions_when_scores_missing():
    metric = StreamingClassificationMetrics()

    metric.update(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1])
    )

    assert np.isclose(metric.auc(), 1.0)


def test_streaming_rolling_accuracy_window():
    metric = StreamingClassificationMetrics(window_size=2)

    metric.update(
        np.array([1, 1, 0]),
        np.array([1, 0, 0])
    )

    assert np.isclose(metric.rolling_accuracy(), 0.5)

    metric.update(
        np.array([1, 0]),
        np.array([0, 1])
    )

    assert np.isclose(metric.rolling_accuracy(), 0.0)


def test_streaming_result_contains_auc_and_rolling_accuracy():
    metric = StreamingClassificationMetrics(window_size=3)

    metric.update(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
        y_score=np.array([0.1, 0.8, 0.3, 0.9])
    )

    result = metric.result()

    assert "accuracy" in result
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert "auc" in result
    assert "confusion_matrix" in result
    assert "rolling_accuracy" in result