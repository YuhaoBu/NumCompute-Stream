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


class StreamingClassificationMetrics:
    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        y_true, y_pred = _check_shape(y_true, y_pred)

        self.tp += int(np.sum((y_true == 1) & (y_pred == 1)))
        self.fp += int(np.sum((y_true == 0) & (y_pred == 1)))
        self.fn += int(np.sum((y_true == 1) & (y_pred == 0)))
        self.tn += int(np.sum((y_true == 0) & (y_pred == 0)))

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def accuracy(self):
        total = self.tp + self.fp + self.fn + self.tn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total

    def precision(self):
        denom = self.tp + self.fp
        if denom == 0:
            return 0.0
        return self.tp / denom

    def recall(self):
        denom = self.tp + self.fn
        if denom == 0:
            return 0.0
        return self.tp / denom

    def f1(self):
        p = self.precision()
        r = self.recall()

        if p + r == 0:
            return 0.0

        return 2 * p * r / (p + r)

    def confusion_matrix(self):
        return np.array([
            [self.tn, self.fp],
            [self.fn, self.tp]
        ])

    def result(self):
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "confusion_matrix": self.confusion_matrix()
        }