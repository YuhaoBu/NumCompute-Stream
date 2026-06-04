import numpy as np


def grad(f, x, h=1e-5, method="central"):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x, dtype=float)

    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'")

    for i in range(x.size):
        x_forward = x.copy()
        x_forward[i] += h

        if method == "forward":
            g[i] = (f(x_forward) - f(x)) / h
        else:
            x_backward = x.copy()
            x_backward[i] -= h
            g[i] = (f(x_forward) - f(x_backward)) / (2 * h)

    return g


def jacobian(F, x, h=1e-5, method="central"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(F(x), dtype=float)

    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'")

    J = np.zeros((y.size, x.size), dtype=float)

    for i in range(x.size):
        x_forward = x.copy()
        x_forward[i] += h

        if method == "forward":
            J[:, i] = (np.asarray(F(x_forward)) - y).ravel() / h
        else:
            x_backward = x.copy()
            x_backward[i] -= h
            J[:, i] = (
                np.asarray(F(x_forward)) - np.asarray(F(x_backward))
            ).ravel() / (2 * h)

    return J

def gradient_descent_step(x, grad, lr=0.01):
    x = np.asarray(x, dtype=float)
    grad = np.asarray(grad, dtype=float)

    if x.shape != grad.shape:
        raise ValueError("Shapes must match.")

    return x - lr * grad