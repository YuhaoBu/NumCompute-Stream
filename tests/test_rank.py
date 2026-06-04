import numpy as np
import pytest

from numcompute.rank import rank, percentile


def test_rank_ordinal():
    arr = np.array([30, 10, 20])
    result = rank(arr, method="ordinal")
    expected = np.array([3.0, 1.0, 2.0])
    assert np.array_equal(result, expected)


def test_rank_dense_with_ties():
    arr = np.array([100, 50, 50, 25])
    result = rank(arr, method="dense")
    expected = np.array([3.0, 2.0, 2.0, 1.0])
    assert np.array_equal(result, expected)


def test_rank_average_with_ties():
    arr = np.array([100, 50, 50, 25])
    result = rank(arr, method="average")
    expected = np.array([4.0, 2.5, 2.5, 1.0])
    assert np.allclose(result, expected)


def test_rank_invalid_method():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        rank(arr, method="wrong")


def test_rank_invalid_dimension():
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        rank(arr)


def test_percentile_single_linear():
    arr = np.array([1, 2, 3, 4, 5])
    result = percentile(arr, 50)
    assert result == 3.0


def test_percentile_multiple():
    arr = np.array([1, 2, 3, 4, 5])
    result = percentile(arr, [0, 50, 100])
    expected = np.array([1.0, 3.0, 5.0])
    assert np.allclose(result, expected)


def test_percentile_lower():
    arr = np.array([1, 2, 3, 4])
    result = percentile(arr, 25, interpolation="lower")
    assert result == 1.0


def test_percentile_higher():
    arr = np.array([1, 2, 3, 4])
    result = percentile(arr, 25, interpolation="higher")
    assert result == 2.0


def test_percentile_midpoint():
    arr = np.array([1, 2, 3, 4])
    result = percentile(arr, 25, interpolation="midpoint")
    assert result == 1.5


def test_percentile_invalid_q():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        percentile(arr, -10)

    with pytest.raises(ValueError):
        percentile(arr, 120)


def test_percentile_invalid_method():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        percentile(arr, 50, interpolation="wrong")