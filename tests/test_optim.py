import numpy as np
import pytest

from numcompute.optim import grad, jacobian


def test_grad_central_quadratic():
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    x = np.array([3.0, 4.0])
    result = grad(f, x, method="central")

    expected = np.array([6.0, 8.0])
    assert np.allclose(result, expected, atol=1e-4)


def test_grad_forward_quadratic():
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    x = np.array([3.0, 4.0])
    result = grad(f, x, method="forward")

    expected = np.array([6.0, 8.0])
    assert np.allclose(result, expected, atol=1e-3)


def test_grad_linear_function():
    def f(x):
        return 2 * x[0] + 3 * x[1]

    x = np.array([1.0, 2.0])
    result = grad(f, x)

    expected = np.array([2.0, 3.0])
    assert np.allclose(result, expected, atol=1e-5)


def test_grad_invalid_method():
    def f(x):
        return x[0] ** 2

    x = np.array([1.0])

    with pytest.raises(ValueError):
        grad(f, x, method="invalid")


def test_jacobian_vector_function():
    def F(x):
        return np.array([
            x[0] + x[1],
            x[0] * x[1]
        ])

    x = np.array([2.0, 3.0])
    result = jacobian(F, x)

    expected = np.array([
        [1.0, 1.0],
        [3.0, 2.0]
    ])

    assert result.shape == (2, 2)
    assert np.allclose(result, expected, atol=1e-4)


def test_jacobian_forward_method():
    def F(x):
        return np.array([
            x[0] ** 2,
            x[1] ** 2
        ])

    x = np.array([3.0, 4.0])
    result = jacobian(F, x, method="forward")

    expected = np.array([
        [6.0, 0.0],
        [0.0, 8.0]
    ])

    assert result.shape == (2, 2)
    assert np.allclose(result, expected, atol=1e-3)


def test_jacobian_invalid_method():
    def F(x):
        return np.array([x[0]])

    x = np.array([1.0])

    with pytest.raises(ValueError):
        jacobian(F, x, method="invalid")
