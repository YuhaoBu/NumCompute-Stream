import numpy as np


def _check_shape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match")
    return y_true, y_pred


def accuracy(y_true, y_pred):
    y_true, y_pred = _check_shape(y_true, y_pred)
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    y_true, y_pred = _check_shape(y_true, y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true, y_pred):
    y_true, y_pred = _check_shape(y_true, y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def confusion_matrix(y_true, y_pred):
    y_true, y_pred = _check_shape(y_true, y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([[tn, fp], [fn, tp]])


def mse(y_true, y_pred):
    y_true, y_pred = _check_shape(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)