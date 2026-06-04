import numpy as np


def _validate_numeric_array(x, name="x"):
    try:
        arr = np.asarray(x, dtype=float)
    except Exception as e:
        raise TypeError(f"{name} must be convertible to a numeric NumPy array.") from e
    return arr


def logsumexp(x, axis=None, keepdims=False):
    arr = _validate_numeric_array(x, name="x")

    max_x = np.max(arr, axis=axis, keepdims=True)
    shifted = arr - max_x
    out = max_x + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)

    if axis is None and not keepdims:
        return float(np.asarray(out))
    return out


def softmax(x, axis=-1):
    arr = _validate_numeric_array(x, name="x")
    lse = logsumexp(arr, axis=axis, keepdims=True)
    return np.exp(arr - lse)


def euclidean_distance(a, b, axis=-1):
    a_arr = _validate_numeric_array(a, name="a")
    b_arr = _validate_numeric_array(b, name="b")

    try:
        diff = a_arr - b_arr
    except ValueError as e:
        raise ValueError("a and b must be broadcast-compatible.") from e

    return np.sqrt(np.sum(diff ** 2, axis=axis))


def batch_iterator(X, batch_size):
    arr = np.asarray(X)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    for start in range(0, len(arr), batch_size):
        yield arr[start:start + batch_size]