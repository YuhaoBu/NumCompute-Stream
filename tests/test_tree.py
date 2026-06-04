import numpy as np
import pytest

from numcompute.tree import DecisionTreeClassifier


def test_decision_tree_fit_predict_simple_binary():
    X = np.array([
        [0.0],
        [0.1],
        [1.0],
        [1.1],
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, y)


def test_decision_tree_partial_fit_multiple_chunks():
    X1 = np.array([
        [0.0],
        [0.1],
    ])
    y1 = np.array([0, 0])

    X2 = np.array([
        [1.0],
        [1.1],
    ])
    y2 = np.array([1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.partial_fit(X1, y1)
    tree.partial_fit(X2, y2)

    X_all = np.vstack([X1, X2])
    y_all = np.concatenate([y1, y2])

    preds = tree.predict(X_all)

    assert np.array_equal(preds, y_all)


def test_decision_tree_predict_before_fit_raises_error():
    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.predict(np.array([[1.0]]))


def test_decision_tree_shape_mismatch_raises_error():
    tree = DecisionTreeClassifier()
    X = np.array([
        [0.0],
        [1.0],
    ])
    y = np.array([0, 1])

    tree.fit(X, y)

    with pytest.raises(ValueError):
        tree.predict(np.array([[1.0, 2.0]]))


def test_decision_tree_handles_nan_values():
    X = np.array([
        [0.0],
        [np.nan],
        [1.0],
        [1.1],
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert preds.shape == y.shape


def test_decision_tree_max_depth_zero_predicts_majority_class():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
    ])
    y = np.array([1, 1, 0])

    tree = DecisionTreeClassifier(max_depth=0)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, np.array([1, 1, 1]))


def test_decision_tree_entropy_criterion_runs():
    X = np.array([
        [0.0],
        [0.1],
        [1.0],
        [1.1],
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2, criterion="entropy")
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, y)


def test_decision_tree_max_features_int_runs():
    X = np.array([
        [0.0, 10.0],
        [0.1, 11.0],
        [1.0, 12.0],
        [1.1, 13.0],
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2, max_features=1)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert preds.shape == y.shape