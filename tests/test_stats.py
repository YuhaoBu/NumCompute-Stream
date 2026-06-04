import numpy as np
import pytest

from numcompute.stats import mean, var, quantile, histogram, Welford


def test_mean_basic():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(mean(x), 2.0)


def test_mean_ignore_nan():
    x = np.array([1.0, np.nan, 3.0])
    assert np.isclose(mean(x, ignore_nan=True), 2.0)


def test_var_basic():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(var(x), np.var(x))


def test_var_ignore_nan():
    x = np.array([1.0, np.nan, 3.0])
    assert np.isclose(var(x, ignore_nan=True), np.nanvar(x))


def test_quantile_median():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(quantile(x, 0.5), 2.5)


def test_quantile_invalid_q():
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        quantile(x, 1.5)


def test_histogram_counts_sum():
    x = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    hist, edges = histogram(x, bins=3)

    assert hist.sum() == len(x)
    assert len(edges) == 4


def test_welford_basic():
    wf = Welford()
    wf.update([1.0, 2.0, 3.0, 4.0])
    result = wf.finalize()

    assert result["count"] == 4
    assert np.isclose(result["mean"], 2.5)
    assert np.isclose(result["variance"], np.var([1.0, 2.0, 3.0, 4.0]))


def test_welford_ignore_nan():
    wf = Welford()
    wf.update([1.0, np.nan, 3.0])
    result = wf.finalize()

    assert result["count"] == 2
    assert np.isclose(result["mean"], 2.0)


def test_welford_empty():
    wf = Welford()
    result = wf.finalize()

    assert result["count"] == 0
    assert np.isnan(result["mean"])
    assert np.isnan(result["variance"])