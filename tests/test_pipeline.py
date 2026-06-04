import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler, MinMaxScaler


def test_pipeline_fit_transform_basic():
    X = np.array([[1.0], [2.0], [3.0]])

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('minmax', MinMaxScaler())
    ])

    X_out = pipe.fit_transform(X)
    assert X_out.shape == X.shape


def test_pipeline_sequential_equivalence():
    X = np.array([[1.0], [2.0], [3.0]])

    scaler = StandardScaler()
    mm = MinMaxScaler()


    X1 = scaler.fit_transform(X)
    X2 = mm.fit_transform(X1)

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('minmax', MinMaxScaler())
    ])

    X_pipe = pipe.fit_transform(X)

    assert np.allclose(X_pipe, X2)


def test_pipeline_transform_after_fit():
    X = np.array([[1.0], [2.0], [3.0]])

    pipe = Pipeline([
        ('scale', StandardScaler())
    ])

    pipe.fit(X)
    X_out = pipe.transform(X)

    assert X_out.shape == X.shape


def test_pipeline_invalid_step():
    class BadStep:
        def fit(self, X):
            return self

    X = np.array([[1.0], [2.0]])

    pipe = Pipeline([
        ('bad', BadStep())
    ])

    with pytest.raises(AttributeError):
        pipe.fit_transform(X)


def test_pipeline_empty_steps():
    X = np.array([[1.0], [2.0]])

    pipe = Pipeline([])

    X_out = pipe.fit_transform(X)

    assert np.array_equal(X_out, X)
