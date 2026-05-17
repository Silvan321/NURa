import numpy as np


def central_difference(f, x, h):
    """
    Second-order central-difference estimate of f'(x).
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def ridders_derivative(
    f,
    x,
    h0=0.1,
    con=1.4,
    tol=1e-10,
    max_orders=20,
    initial_capacity=8,
    safe_factor=2.0,
):
    """
    Estimate f'(x) using Ridders' method.

    Parameters
    ----------
    f : callable
        Function f(x).
    x : float
        Point where derivative is evaluated.
    h0 : float
        Initial step size.
    con : float
        Step-size reduction factor. Each new h is h / con.
    tol : float
        Desired absolute error tolerance.
    max_orders : int
        Maximum number of extrapolation orders.
    initial_capacity : int
        Initial size of the extrapolation vector.
    safe_factor : float
        Stop early if error grows too much compared to best error.

    Returns
    -------
    derivative : float
        Best derivative estimate found.
    error : float
        Estimated absolute error.
    orders_used : int
        Number of raw central-difference estimates used.
    """

    if h0 <= 0:
        raise ValueError("h0 must be positive")

    if con <= 1:
        raise ValueError("con must be greater than 1")

    if max_orders < 1:
        raise ValueError("max_orders must be at least 1")

    capacity = max(1, initial_capacity)
    a = np.empty(capacity, dtype=float)

    best = np.nan
    best_err = np.inf

    h = h0

    for i in range(max_orders):
        # Grow vector if needed.
        if i >= len(a):
            new_a = np.empty(2 * len(a), dtype=a.dtype)
            new_a[:i] = a[:i]
            a = new_a

        # First column of the extrapolation tableau:
        # raw central-difference estimate at step h.
        a[i] = central_difference(f, x, h)

        # Richardson/Ridders extrapolation.
        #
        # After this loop:
        #   a[0] is the highest-order estimate using rows 0..i.
        #   a[1] is the next lower extrapolated estimate.
        #   ...
        #   a[i] is the raw estimate for the newest h.
        for j in range(i - 1, -1, -1):
            factor = con ** (2 * (i - j))
            a[j] = (factor * a[j + 1] - a[j]) / (factor - 1.0)

        current = a[0]

        # Estimate error by comparing successive best extrapolated values.
        if i == 0:
            best = current
        else:
            err = abs(current - best)

            if err < best_err:
                best_err = err
                best = current

            if err < tol:
                return best, best_err, i + 1

            # Optional safety stop:
            # if things are getting worse after already having a good estimate.
            if err > safe_factor * best_err:
                return best, best_err, i + 1

        h /= con

    return best, best_err, max_orders


def f(x):
    return np.sin(x)


d, err, orders = ridders_derivative(f, x=1.0)

print(d)
print(np.cos(1.0))
print(err)
print(orders)
