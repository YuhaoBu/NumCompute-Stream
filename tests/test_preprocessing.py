import numpy as np
import pytest

from numcompute.preprocessing import StandardScaler, MinMaxScaler, Imputer, OneHotEncoder


def test_standard_scaler_basic():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    expected = np.array([[-1.0, -1.0], [1.0, 1.0]])
    assert X_scaled.shape == X.shape
    assert np.allclose(X_scaled, expected)


def test_standard_scaler_constant_column():
    X = np.array([[5.0, 1.0], [5.0, 3.0], [5.0, 5.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled[:, 0], np.array([0.0, 0.0, 0.0]))
    assert X_scaled.shape == X.shape


def test_standard_scaler_transform_before_fit():
    scaler = StandardScaler()
    X = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        scaler.transform(X)


def test_minmax_scaler_basic():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    assert np.allclose(X_scaled, expected)


def test_imputer_mean():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    imputer = Imputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    expected = np.array([[1.0, 5.0], [3.0, 4.0], [5.0, 6.0]])
    assert np.allclose(X_imputed, expected)


def test_imputer_median():
    X = np.array([[1.0, np.nan], [3.0, 100.0], [5.0, 6.0]])
    imputer = Imputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    expected = np.array([[1.0, 53.0], [3.0, 100.0], [5.0, 6.0]])
    assert np.allclose(X_imputed, expected)


def test_imputer_most_frequent():
    X = np.array([[1.0, np.nan], [1.0, 2.0], [3.0, 2.0]])
    imputer = Imputer(strategy="most_frequent")
    X_imputed = imputer.fit_transform(X)

    expected = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 2.0]])
    assert np.allclose(X_imputed, expected)


def test_one_hot_encoder_basic():
    x = np.array(["red", "blue", "red"])
    encoder = OneHotEncoder()
    out = encoder.fit_transform(x)

    assert out.shape == (3, 2)
    assert np.array_equal(out[0], out[2])
    assert not np.array_equal(out[0], out[1])


def test_one_hot_encoder_unknown_error():
    x_train = np.array(["red", "blue"])
    x_test = np.array(["red", "green"])

    encoder = OneHotEncoder(handle_unknown="error")
    encoder.fit(x_train)

    with pytest.raises(ValueError):
        encoder.transform(x_test)


def test_one_hot_encoder_unknown_ignore():
    x_train = np.array(["red", "blue"])
    x_test = np.array(["red", "green"])

    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(x_train)
    out = encoder.transform(x_test)

    assert out.shape == (2, 2)
    assert np.array_equal(out[1], np.array([0, 0]))