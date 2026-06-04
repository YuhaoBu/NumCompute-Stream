import numpy as np


def _validate_numeric_array(x, name="x"):
    try:
        arr = np.asarray(x, dtype=float)
    except Exception as e:
        raise TypeError(f"{name} must be convertible to a numeric NumPy array.") from e
    return arr


def mean(x, axis=None, ignore_nan=True, keepdims=False):
    arr = _validate_numeric_array(x, name="x")
    if ignore_nan:
        return np.nanmean(arr, axis=axis, keepdims=keepdims)
    return np.mean(arr, axis=axis, keepdims=keepdims)


def var(x, axis=None, ddof=0, ignore_nan=True, keepdims=False):
    arr = _validate_numeric_array(x, name="x")
    if ignore_nan:
        return np.nanvar(arr, axis=axis, ddof=ddof, keepdims=keepdims)
    return np.var(arr, axis=axis, ddof=ddof, keepdims=keepdims)


def histogram(x, bins=10, range=None):
    arr = _validate_numeric_array(x, name="x").ravel()
    arr = arr[~np.isnan(arr)]
    return np.histogram(arr, bins=bins, range=range)


def quantile(x, q, axis=None, ignore_nan=True, keepdims=False):
    q_arr = np.asarray(q, dtype=float)
    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be in the interval [0, 1].")

    arr = _validate_numeric_array(x, name="x")
    if ignore_nan:
        return np.nanquantile(arr, q_arr, axis=axis, keepdims=keepdims)
    return np.quantile(arr, q_arr, axis=axis, keepdims=keepdims)


class Welford:
    def __init__(self):
        self.n = 0
        self.mean_ = 0.0
        self.M2 = 0.0

    def update(self, x):
        values = np.asarray(x, dtype=float).ravel()
        values = values[~np.isnan(values)]

        for value in values:
            self.n += 1
            delta = value - self.mean_
            self.mean_ += delta / self.n
            delta2 = value - self.mean_
            self.M2 += delta * delta2

        return self

    def finalize(self, ddof=0):
        if self.n == 0:
            return {"count": 0, "mean": np.nan, "variance": np.nan}
        if self.n - ddof <= 0:
            raise ValueError("ddof must be smaller than the number of valid observations.")

        variance = self.M2 / (self.n - ddof)
        return {"count": self.n, "mean": self.mean_, "variance": variance}

class StreamingStats:
    def __init__(self, bins=10, hist_range=None):
        self.count_ = None
        self.mean_ = None
        self.M2_ = None
        self.n_features_in_ = None
        self.bins = bins
        self.hist_range = hist_range
        self._data = None

    def update_stats(self, X_chunk):
        X = _validate_numeric_array(X_chunk, name="X_chunk")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(f"X_chunk must be 1D or 2D, got shape {X.shape}.")

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(X.shape[1], dtype=int)
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.M2_ = np.zeros(X.shape[1], dtype=float)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_chunk has {X.shape[1]} features, but StreamingStats was initialized "
                f"with {self.n_features_in_} features."
            )

        valid = ~np.isnan(X)
        chunk_count = np.sum(valid, axis=0)

        if np.any(chunk_count > 0):
            chunk_mean = np.zeros(X.shape[1], dtype=float)
            chunk_M2 = np.zeros(X.shape[1], dtype=float)

            cols = chunk_count > 0
            chunk_mean[cols] = np.nanmean(X[:, cols], axis=0)
            chunk_var = np.nanvar(X[:, cols], axis=0)
            chunk_M2[cols] = chunk_var * chunk_count[cols]

            old_count = self.count_.copy()
            new_count = old_count + chunk_count

            update_cols = chunk_count > 0
            delta = chunk_mean[update_cols] - self.mean_[update_cols]

            total = new_count[update_cols]
            old = old_count[update_cols]
            new = chunk_count[update_cols]

            self.mean_[update_cols] = (
                self.mean_[update_cols] + delta * new / total
            )

            self.M2_[update_cols] = (
                self.M2_[update_cols]
                + chunk_M2[update_cols]
                + (delta ** 2) * old * new / total
            )

            self.count_ = new_count

        if self._data is None:
            self._data = X.copy()
        else:
            self._data = np.vstack([self._data, X])

        return self

    def mean(self):
        if self.count_ is None:
            return np.nan
        return np.where(self.count_ > 0, self.mean_, np.nan)

    def variance(self, ddof=0):
        if self.count_ is None:
            return np.nan

        out = np.full_like(self.mean_, np.nan, dtype=float)
        valid = self.count_ > ddof
        out[valid] = self.M2_[valid] / (self.count_[valid] - ddof)
        return out

    def quantile(self, q):
        if self._data is None:
            return np.nan

        q_arr = np.asarray(q, dtype=float)
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("q must be in the interval [0, 1].")

        return np.nanquantile(self._data, q_arr, axis=0)

    def histogram(self, bins=None, hist_range=None):
        if self._data is None:
            return []

        bins = self.bins if bins is None else bins
        hist_range = self.hist_range if hist_range is None else hist_range

        results = []
        for j in range(self.n_features_in_):
            col = self._data[:, j]
            col = col[~np.isnan(col)]
            results.append(np.histogram(col, bins=bins, range=hist_range))

        return results

    def result(self):
        return {
            "count": self.count_,
            "mean": self.mean(),
            "variance": self.variance(),
            "histogram": self.histogram(),
        }