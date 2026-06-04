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