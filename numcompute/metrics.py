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
    """
    Accumulate classification metrics over streaming data chunks.
    Supports cumulative accuracy, precision, recall, F1 score,
    confusion matrix, AUC, and rolling-window accuracy.
    """
    def __init__(self, window_size=None):
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be a positive integer or None.")

        self.window_size = window_size
        self.reset()

    def update(self, y_true, y_pred, y_score=None):
        y_true, y_pred = _check_shape(y_true, y_pred)

        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

        if y_score is None:
            y_score = y_pred.astype(float)
        else:
            y_score = np.asarray(y_score, dtype=float)
            if y_score.shape != y_true.shape:
                raise ValueError("Shapes of y_true and y_score must match")
            y_score = y_score.ravel()

        self.tp += int(np.sum((y_true == 1) & (y_pred == 1)))
        self.fp += int(np.sum((y_true == 0) & (y_pred == 1)))
        self.fn += int(np.sum((y_true == 1) & (y_pred == 0)))
        self.tn += int(np.sum((y_true == 0) & (y_pred == 0)))

        self._y_true_history.append(y_true.copy())
        self._score_history.append(y_score.copy())

        self._update_rolling_window(y_true, y_pred)

        return self

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

        self._y_true_history = []
        self._score_history = []

        self._rolling_y_true = np.array([], dtype=int)
        self._rolling_y_pred = np.array([], dtype=int)

        return self

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

    def auc(self):
        if len(self._y_true_history) == 0:
            return 0.0

        y_true = np.concatenate(self._y_true_history)
        y_score = np.concatenate(self._score_history)

        positive = y_true == 1
        negative = y_true == 0

        n_pos = int(np.sum(positive))
        n_neg = int(np.sum(negative))

        if n_pos == 0 or n_neg == 0:
            return 0.0

        ranks = self._average_ranks(y_score)
        sum_pos_ranks = np.sum(ranks[positive])

        auc_value = (
            sum_pos_ranks - n_pos * (n_pos + 1) / 2
        ) / (n_pos * n_neg)

        return float(auc_value)

    def rolling_accuracy(self):
        if self.window_size is None:
            return self.accuracy()

        if self._rolling_y_true.size == 0:
            return 0.0

        return float(np.mean(self._rolling_y_true == self._rolling_y_pred))

    def rolling_result(self):
        return {
            "rolling_accuracy": self.rolling_accuracy(),
            "window_size": self.window_size,
        }

    def result(self):
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "auc": self.auc(),
            "confusion_matrix": self.confusion_matrix(),
            "rolling_accuracy": self.rolling_accuracy(),
        }

    def _update_rolling_window(self, y_true, y_pred):
        if self.window_size is None:
            return

        self._rolling_y_true = np.concatenate([
            self._rolling_y_true,
            y_true.astype(int)
        ])

        self._rolling_y_pred = np.concatenate([
            self._rolling_y_pred,
            y_pred.astype(int)
        ])

        if self._rolling_y_true.size > self.window_size:
            self._rolling_y_true = self._rolling_y_true[-self.window_size:]
            self._rolling_y_pred = self._rolling_y_pred[-self.window_size:]

    def _average_ranks(self, scores):
        scores = np.asarray(scores, dtype=float)

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]

        unique_scores, inverse, counts = np.unique(
            sorted_scores,
            return_inverse=True,
            return_counts=True
        )

        end_ranks = np.cumsum(counts)
        start_ranks = end_ranks - counts + 1
        average_ranks = (start_ranks + end_ranks) / 2.0

        sorted_ranks = average_ranks[inverse]

        ranks = np.empty_like(sorted_ranks, dtype=float)
        ranks[order] = sorted_ranks

        return ranks