"""
rank.py

Ranking and percentile utilities built with NumPy.
"""

from __future__ import annotations

import numpy as np


def _validate_1d_array(values):
    arr = np.asarray(values)

    if arr.size == 0:
        raise ValueError("Input array must not be empty.")

    if arr.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    return arr


def rank(data, method="average"):
    """
    Rank 1D data with tie handling.

    Parameters
    ----------
    data : array-like
        Input 1D data.
    method : {"average", "dense", "ordinal"}, default="average"
        Ranking method.

    Returns
    -------
    np.ndarray
        Array of ranks as float for average, int-compatible for others.

    Raises
    ------
    ValueError
        If input is invalid or method is unsupported.
    """
    arr = _validate_1d_array(data)

    if method not in {"average", "dense", "ordinal"}:
        raise ValueError("method must be one of: 'average', 'dense', 'ordinal'.")

    # Stable argsort keeps original order for ties
    order = np.argsort(arr, kind="stable")
    sorted_arr = arr[order]

    n = arr.size
    ranks = np.empty(n, dtype=float)

    if method == "ordinal":
        ranks[order] = np.arange(1, n + 1, dtype=float)
        return ranks

    unique_vals, first_idx, counts = np.unique(
        sorted_arr, return_index=True, return_counts=True
    )

    if method == "dense":
        for dense_rank, (start, count) in enumerate(zip(first_idx, counts), start=1):
            group_indices = order[start:start + count]
            ranks[group_indices] = float(dense_rank)
        return ranks

    # average
    for start, count in zip(first_idx, counts):
        avg_rank = (start + 1 + start + count) / 2.0
        group_indices = order[start:start + count]
        ranks[group_indices] = avg_rank

    return ranks


def percentile(data, q, interpolation="linear"):
    """
    Compute percentile values for 1D data.

    Parameters
    ----------
    data : array-like
        Input 1D data.
    q : float or array-like
        Percentile or percentiles in the range [0, 100].
    interpolation : {"linear", "lower", "higher", "midpoint"}, default="linear"
        Interpolation method.

    Returns
    -------
    scalar or np.ndarray
        Percentile value(s).

    Raises
    ------
    ValueError
        If input is invalid.
    """
    arr = _validate_1d_array(data)

    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError(
            "interpolation must be one of: 'linear', 'lower', 'higher', 'midpoint'."
        )

    q_arr = np.asarray(q, dtype=float)

    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be in the range [0, 100].")

    arr_sorted = np.sort(arr)

    def _single_percentile(qv):
        if arr_sorted.size == 1:
            return float(arr_sorted[0])

        pos = (qv / 100.0) * (arr_sorted.size - 1)
        lower_idx = int(np.floor(pos))
        upper_idx = int(np.ceil(pos))

        lower_val = arr_sorted[lower_idx]
        upper_val = arr_sorted[upper_idx]

        if interpolation == "lower":
            return float(lower_val)
        if interpolation == "higher":
            return float(upper_val)
        if interpolation == "midpoint":
            return float((lower_val + upper_val) / 2.0)

        # linear
        if lower_idx == upper_idx:
            return float(arr_sorted[lower_idx])

        weight = pos - lower_idx
        return float(lower_val + weight * (upper_val - lower_val))

    if q_arr.ndim == 0:
        return _single_percentile(float(q_arr))

    return np.array([_single_percentile(val) for val in q_arr], dtype=float)