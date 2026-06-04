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

class DummyStreamingModel:
    def __init__(self):
        self.was_partial_fit_called = False
        self.X_shape_ = None
        self.y_shape_ = None

    def partial_fit(self, X, y=None):
        self.was_partial_fit_called = True
        self.X_shape_ = X.shape
        if y is not None:
            self.y_shape_ = y.shape
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


def test_pipeline_partial_fit_with_transformer_and_model():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DummyStreamingModel())
    ])

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    y = np.array([0, 1, 0])

    pipe.partial_fit(X, y)

    model = pipe.steps[-1][1]

    assert model.was_partial_fit_called is True
    assert model.X_shape_ == X.shape
    assert model.y_shape_ == y.shape


def test_pipeline_predict_after_partial_fit():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DummyStreamingModel())
    ])

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    y = np.array([0, 1])

    pipe.partial_fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (2,)


def test_pipeline_requires_partial_fit_for_transformer():
    class BadTransformer:
        def transform(self, X):
            return X

    pipe = Pipeline([
        ("bad", BadTransformer()),
        ("model", DummyStreamingModel())
    ])

    X = np.array([[1.0, 2.0]])
    y = np.array([0])

    with pytest.raises(AttributeError):
        pipe.partial_fit(X, y)


def test_pipeline_requires_partial_fit_for_final_model():
    class BadModel:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", BadModel())
    ])

    X = np.array([[1.0, 2.0]])
    y = np.array([0])

    with pytest.raises(AttributeError):
        pipe.partial_fit(X, y)


def test_pipeline_requires_predict_for_final_model():
    class NoPredictModel:
        def partial_fit(self, X, y=None):
            return self

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", NoPredictModel())
    ])

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    y = np.array([0, 1])

    pipe.partial_fit(X, y)

    with pytest.raises(AttributeError):
        pipe.predict(X)
