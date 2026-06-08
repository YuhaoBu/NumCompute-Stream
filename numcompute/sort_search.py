"""
sort_search.py

Sorting and searching utilities built with NumPy.

This module provides:
- stable_sort
- multi_key_sort
- topk
- quickselect
- binary_search

All functions are designed for educational use in the NumCompute project.
"""

from __future__ import annotations

import numpy as np


def _validate_array(values):
    """
    Convert input to a NumPy array and check it is not empty.

    Parameters
    ----------
    values : array-like
        Input data.

    Returns
    -------
    np.ndarray
        Input converted to NumPy array.

    Raises
    ------
    ValueError
        If the array is empty.
    """
    arr = np.asarray(values)
    if arr.size == 0:
        raise ValueError("Input array must not be empty.")
    return arr


def stable_sort(values, axis=-1, ascending=True):
    """
    Perform a stable sort using NumPy.

    Parameters
    ----------
    values : array-like
        Input array to sort.
    axis : int, default=-1
        Axis along which to sort.
    ascending : bool, default=True
        If True, sort in ascending order.
        If False, sort in descending order.

    Returns
    -------
    np.ndarray
        Sorted array.

    Raises
    ------
    ValueError
        If axis is invalid.
    """
    arr = _validate_array(values)

    if axis >= arr.ndim or axis < -arr.ndim:
        raise ValueError("Invalid axis for input array.")

    sorted_arr = np.sort(arr, axis=axis, kind="stable")

    if not ascending:
        sorted_arr = np.flip(sorted_arr, axis=axis)

    return sorted_arr


def multi_key_sort(values, keys=None, ascending=True):
    """
    Sort a 2D array by one or more columns.

    Parameters
    ----------
    values : array-like
        A 2D array.
    keys : list[int] or tuple[int], optional
        Column indices used for sorting.
        If None, all columns are used from left to right.
    ascending : bool or list[bool], default=True
        Sort direction(s). If a single bool is given, it applies to all keys.
        If a list is given, it must match the number of keys.

    Returns
    -------
    np.ndarray
        Sorted 2D array.

    Raises
    ------
    ValueError
        If input is not 2D, or keys are invalid.
    """
    arr = np.asarray(values)

    if arr.ndim != 2:
        raise ValueError("multi_key_sort expects a 2D array.")

    n_rows, n_cols = arr.shape

    if n_rows == 0:
        raise ValueError("Input array must not be empty.")

    if keys is None:
        keys = list(range(n_cols))

    if len(keys) == 0:
        raise ValueError("keys must not be empty.")

    for key in keys:
        if key < 0 or key >= n_cols:
            raise ValueError(f"Invalid column index: {key}")

    if isinstance(ascending, bool):
        ascending = [ascending] * len(keys)

    if len(ascending) != len(keys):
        raise ValueError("Length of ascending must match length of keys.")

    # np.lexsort uses the last key as the primary key,
    # so we prepare keys in reverse order.
    sort_keys = []
    for col, asc in zip(keys, ascending):
        column_data = arr[:, col]
        if asc:
            sort_keys.append(column_data)
        else:
            sort_keys.append(-column_data)

    indices = np.lexsort(tuple(sort_keys[::-1]))
    return arr[indices]


def topk(values, k, largest=True, return_indices=True):
    """
    Return the top-k values using np.argpartition.

    Parameters
    ----------
    values : array-like
        1D input array.
    k : int
        Number of top elements to return.
    largest : bool, default=True
        If True, return largest k elements.
        If False, return smallest k elements.
    return_indices : bool, default=True
        If True, return both values and indices.
        If False, return only values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or np.ndarray
        Top-k values and their indices, or only values.

    Raises
    ------
    ValueError
        If input is not 1D or k is invalid.
    """
    arr = _validate_array(values)

    if arr.ndim != 1:
        raise ValueError("topk expects a 1D array.")

    if not isinstance(k, int) or k <= 0 or k > arr.size:
        raise ValueError("k must be an integer between 1 and len(values).")

    if largest:
        partition_idx = np.argpartition(arr, -k)[-k:]
        selected_values = arr[partition_idx]
        order = np.argsort(selected_values)[::-1]
    else:
        partition_idx = np.argpartition(arr, k - 1)[:k]
        selected_values = arr[partition_idx]
        order = np.argsort(selected_values)

    final_indices = partition_idx[order]
    final_values = arr[final_indices]

    if return_indices:
        return final_values, final_indices
    return final_values


def quickselect(values, k, largest=False):
    """
    Return the k-th selected value for educational purposes.

    This implementation uses NumPy partition to simulate quickselect behavior.

    Parameters
    ----------
    values : array-like
        1D input array.
    k : int
        Zero-based index.
    largest : bool, default=False
        If False, return the k-th smallest value.
        If True, return the k-th largest value.

    Returns
    -------
    scalar
        The selected value.

    Raises
    ------
    ValueError
        If input is not 1D or k is out of range.
    """
    arr = _validate_array(values)

    if arr.ndim != 1:
        raise ValueError("quickselect expects a 1D array.")

    if not isinstance(k, int) or k < 0 or k >= arr.size:
        raise ValueError("k must be in the range [0, len(values)-1].")

    if largest:
        target_index = arr.size - 1 - k
    else:
        target_index = k

    partitioned = np.partition(arr, target_index)
    return partitioned[target_index]


def binary_search(sorted_array, x):
    """
    Perform binary search on a sorted 1D array.

    Returns insertion index and existence flag.

    Parameters
    ----------
    sorted_array : array-like
        Sorted 1D array.
    x : scalar
        Value to search for.

    Returns
    -------
    tuple[int, bool]
        insertion_index, found

    Raises
    ------
    ValueError
        If input is not 1D or is empty.
    """
    arr = _validate_array(sorted_array)

    if arr.ndim != 1:
        raise ValueError("binary_search expects a 1D sorted array.")

    idx = int(np.searchsorted(arr, x, side="left"))
    found = idx < arr.size and arr[idx] == x
    return idx, bool(found)