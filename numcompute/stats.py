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