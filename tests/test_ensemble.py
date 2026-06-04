import numpy as np
import pytest

from numcompute.ensemble import EnsembleClassifier


def test_ensemble_partial_fit_predict_shape():
    X = np.array([
        [0.0],
        [0.1],
        [1.0],
        [1.1],
    ])
    y = np.array([0, 0, 1, 1])

    model = EnsembleClassifier(
        n_estimators=3,
        max_depth=2,
        random_state=42,
    )

    model.partial_fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape


def test_ensemble_partial_fit_multiple_chunks():
    X1 = np.array([
        [0.0],
        [0.1],
    ])
    y1 = np.array([0, 0])

    X2 = np.array([
        [1.0],
        [1.1],
    ])
    y2 = np.array([1, 1])

    model = EnsembleClassifier(
        n_estimators=3,
        max_depth=2,
        random_state=42,
    )

    model.partial_fit(X1, y1)
    model.partial_fit(X2, y2)

    X_all = np.vstack([X1, X2])
    preds = model.predict(X_all)

    assert preds.shape == (4,)


def test_ensemble_n_estimators_created():
    model = EnsembleClassifier(n_estimators=7)

    assert len(model.estimators_) == 7


def test_ensemble_predict_before_fit_raises_error():
    model = EnsembleClassifier(n_estimators=3)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0]]))


def test_ensemble_shape_mismatch_raises_error():
    X = np.array([
        [0.0],
        [1.0],
    ])
    y = np.array([0, 1])

    model = EnsembleClassifier(n_estimators=3)
    model.partial_fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0]]))


def test_ensemble_invalid_n_estimators():
    with pytest.raises(ValueError):
        EnsembleClassifier(n_estimators=0)


def test_ensemble_invalid_method():
    with pytest.raises(ValueError):
        EnsembleClassifier(method="boosting")