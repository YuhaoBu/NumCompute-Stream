import sys
import numpy as np

from numcompute.metrics import StreamingClassificationMetrics, accuracy


class StreamTrainer:
    def __init__(self, model):
        self.model = model
        self.metrics = StreamingClassificationMetrics()
        self.logs_ = []
        self.fit_chunks_ = 0
        self.score_chunks_ = 0

    def fit_chunk(self, X, y):
        X, y = self._validate_X_y(X, y)

        if not hasattr(self.model, "partial_fit"):
            raise AttributeError("model must have a partial_fit method.")

        self.model.partial_fit(X, y)
        self.fit_chunks_ += 1

        return self

    def score_chunk(self, X, y):
        X, y = self._validate_X_y(X, y)

        if not hasattr(self.model, "predict"):
            raise AttributeError("model must have a predict method.")

        y_pred = self.model.predict(X)

        if y_pred.shape != y.shape:
            raise ValueError("Predicted labels must have the same shape as y.")

        self.metrics.update(y, y_pred)

        chunk_accuracy = accuracy(y, y_pred)
        cumulative_result = self.metrics.result()

        self.score_chunks_ += 1

        log = {
            "chunk": self.score_chunks_,
            "chunk_accuracy": float(chunk_accuracy),
            "chunk_error": float(1.0 - chunk_accuracy),
            "cumulative_accuracy": float(cumulative_result["accuracy"]),
            "memory_bytes": self._memory_footprint(X, y, y_pred),
        }

        self.logs_.append(log)
        return log

    def fit_score_chunk(self, X, y):
        self.fit_chunk(X, y)
        return self.score_chunk(X, y)

    def get_logs(self):
        return list(self.logs_)

    def reset_logs(self):
        self.logs_ = []
        self.metrics.reset()
        self.score_chunks_ = 0
        return self

    def _validate_X_y(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")

        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got shape {y.shape}.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if X.shape[0] == 0:
            raise ValueError("X and y must not be empty.")

        return X, y

    def _memory_footprint(self, X, y, y_pred):
        total = sys.getsizeof(self.model)
        total += np.asarray(X).nbytes
        total += np.asarray(y).nbytes
        total += np.asarray(y_pred).nbytes
        return int(total)