import numpy as np
import pytest

from numcompute.sort_search import (
    stable_sort,
    multi_key_sort,
    topk,
    quickselect,
    binary_search,
)


# =====================
# stable_sort
# =====================
def test_stable_sort_basic():
    arr = np.array([3, 1, 2])
    result = stable_sort(arr)
    assert np.array_equal(result, np.array([1, 2, 3]))


def test_stable_sort_descending():
    arr = np.array([3, 1, 2])
    result = stable_sort(arr, ascending=False)
    assert np.array_equal(result, np.array([3, 2, 1]))


# =====================
# multi_key_sort
# =====================
def test_multi_key_sort_basic():
    arr = np.array([
        [2, 30],
        [1, 20],
        [2, 10],
    ])
    result = multi_key_sort(arr, keys=[0, 1], ascending=[True, True])
    expected = np.array([
        [1, 20],
        [2, 10],
        [2, 30],
    ])
    assert np.array_equal(result, expected)


# =====================
# topk
# =====================
def test_topk_largest():
    arr = np.array([5, 1, 9, 3, 7])
    values, indices = topk(arr, 2)
    assert set(values) == {9, 7}


def test_topk_smallest():
    arr = np.array([5, 1, 9, 3, 7])
    values, indices = topk(arr, 2, largest=False)
    assert set(values) == {1, 3}


# =====================
# quickselect
# =====================
def test_quickselect_smallest():
    arr = np.array([5, 1, 9, 3, 7])
    result = quickselect(arr, 2)
    assert result == 5  # 第3小


def test_quickselect_largest():
    arr = np.array([5, 1, 9, 3, 7])
    result = quickselect(arr, 1, largest=True)
    assert result == 7  # 第2大


# =====================
# binary_search
# =====================
def test_binary_search_found():
    arr = np.array([1, 3, 5, 7, 9])
    idx, found = binary_search(arr, 5)
    assert found is True
    assert idx == 2


def test_binary_search_not_found():
    arr = np.array([1, 3, 5, 7, 9])
    idx, found = binary_search(arr, 4)
    assert found is False
    assert idx == 2

def test_topk_largest_sorted_output():
    arr = np.array([5, 1, 9, 3, 7])
    values, indices = topk(arr, 3)
    assert np.array_equal(values, np.array([9, 7, 5]))


def test_topk_smallest_sorted_output():
    arr = np.array([5, 1, 9, 3, 7])
    values, indices = topk(arr, 3, largest=False)
    assert np.array_equal(values, np.array([1, 3, 5]))


def test_topk_return_values_only():
    arr = np.array([5, 1, 9, 3, 7])
    values = topk(arr, 2, return_indices=False)
    assert np.array_equal(values, np.array([9, 7]))


def test_topk_k_equals_length():
    arr = np.array([4, 2, 8])
    values, indices = topk(arr, 3)
    assert np.array_equal(values, np.array([8, 4, 2]))


def test_topk_invalid_k_zero():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        topk(arr, 0)


def test_topk_invalid_k_too_large():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        topk(arr, 4)


def test_topk_invalid_input_dimension():
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        topk(arr, 2)

def test_quickselect_basic():
    arr = np.array([5, 1, 9, 3, 7])
    assert quickselect(arr, 0) == 1
    assert quickselect(arr, 1) == 3
    assert quickselect(arr, 2) == 5


def test_quickselect_last():
    arr = np.array([5, 1, 9, 3, 7])
    assert quickselect(arr, 4) == 9


def test_quickselect_invalid_k():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        quickselect(arr, -1)

    with pytest.raises(ValueError):
        quickselect(arr, 3)