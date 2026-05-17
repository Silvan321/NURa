from collections.abc import Callable
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


def func(x):
    return x**2 * np.sin(x)


def func_derivative(x):
    return 2 * x * np.sin(x) + x**2 * np.cos(x)


def central_difference(func: Callable, h: float, x: np.ndarray):
    return (func(x + h) - func(x - h)) / (2 * h)


def ridders(func: Callable, h: float, x: np.ndarray, d: float = 2, m: int = 5, relative_target_error: float = (np.finfo(float).eps) ** 0.8):  # noqa: RET503
    """Use Ridder's method (Richardson extrapolation for differentiation) to compute the numerical derivative of a function func,
    using the central differences formula with an initial h.
    We return the best estimate of the derivative, the estimated error, and the number of approximations used.

    d is the factor to decrease h with for successive approximations.
    m is the number of initial approximations.
    We will continue improving our estimate until the best estimate is smaller than the target error.
    The default target error (a relative error) is the machine epsilon to the power 0.7, based on slide 9 of lecture 5.
    To get the absolute error, we can multiply the relative error with the calculated value of the derivative at the point of interest.
    We also add an absolute target error, to make sure we can also get good estimates for small derivatives, where the relative error can be large even if the absolute error is small.
    """
    absolute_target_error = 10 * np.finfo(float).eps
    safety_factor = 2  # Just like the Numerical Recipes book, we take a safety factor, so that the error can grow slightly after already having a good estimate, but we stop if it grows too much, which is a sign that we are starting to get into the regime where numerical errors dominate.
    d_inv = 1 / d  # calculate this once for efficiency
    derivative_vector = np.zeros(m)

    best_derivative = np.nan
    best_error = np.inf

    i = 0
    while i < derivative_vector.size + 1:
        derivative_vector[i] = central_difference(func, h, x)
        for j in range(i - 1, -1, -1):
            factor = d ** (2 * (i - j))  # Combine pairs of approximations backwards, AS SOON AS THEY ARE AVAILABLE, so that we can stop early if the error starts to grow!
            derivative_vector[j] = (factor * derivative_vector[j + 1] - derivative_vector[j]) / (
                factor - 1
            )  # Combine pairs of approximations (step 3 of slide 8), but with update rule adapted for 1D version (overwriting of previous approximations)
        h *= d_inv

        current = derivative_vector[0]

        # Estimate error by comparing successive best extrapolated values.
        if i == 0:
            best_derivative = current
        else:
            error = abs(current - best_derivative)  # The error is the difference between approximations

            if error < best_error:  # the difference between approximations should become smaller
                best_error = error
                best_derivative = current

            if error < absolute_target_error + relative_target_error * abs(current):
                return best_derivative, best_error, i + 1

            # if things are getting worse after already having a good estimate.
            if error > safety_factor * best_error:
                return best_derivative, best_error, i + 1

        i += 1
        if i == derivative_vector.size:  # Increase derivative vector if needed
            derivative_vector_new = np.zeros(2 * derivative_vector.size)
            derivative_vector_new[: derivative_vector.size] = derivative_vector
            derivative_vector = derivative_vector_new


def ridder_array_method(func: Callable, h: float, x: np.ndarray, d: float = 2, m: int = 5, target_error: float = (np.finfo(float).eps) ** 0.8):  # noqa: RET503
    """Use Ridder's method (Richardson extrapolation for differentiation) to compute the numerical derivative of a function func,
    using the central differences formula with an initial h.
    d is the factor to decrease h with for successive approximations.
    m is the number of initial approximations.
    This approach is based on slide 8 of lecture 5, in such a way that it can be called with an x array at once.
    We will continue improving our estimate until the best estimate is smaller than the target error.
    The default target error is the machine relative error to the power 0.8, based on slide 9 of lecture 5.
    """
    D = np.zeros((m, len(x)))  # matrix storing m solution approximations for each x value in the input x array

    d_inv = 1 / d  # calculate this once for efficiency
    h_vector = np.zeros(m)
    h_vector[0] = h  # for plotting purposes, remove when we want only efficiency.
    improvement = 2 * target_error  # make sure while loop starts
    for j in range(m):
        for i in range(m - j):
            best_approx = deepcopy(D[0])
            if j == 0 and i == 0:
                D[0] = central_difference(func, h, x)
            elif j == 0:
                h *= d_inv
                h_vector[i] = h
                D[i] = central_difference(func, h, x)
            else:
                # Now combine pairs of approximations (step 3 of slide 8), but with update rule adapted for 1D version (overwriting of previous approximations)
                D[i] = (d ** (2 * j) * D[i + 1] - D[i]) / (d ** (2 * j) - 1)
            improvement = abs(best_approx - D[0])
            if improvement < target_error:  # Terminate when improvement over previous best approx is smaller than target error
                return D[0], h_vector
            if improvement > previous_improvement:  # How can I measure if error grows? like this?
                return best_approx
            previous_improvement = improvement


def ridders_derivative_generated(
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


def main():

    # Assignment 1a
    x = np.linspace(0, 2 * np.pi, num=2000)

    # Assignment 1b
    # Use the central difference method to calculate the numerical derivative for different values of the step size h
    # It turns out the one with the largest step size (h=1) resembles the analytical derivative the best
    fig, axs = plt.subplots(1, 4)
    plt.suptitle("Central difference method for numerical differentiation")
    h_list = [1, 0.1, 0.01, 0.001]
    for i, h in enumerate(h_list):
        axs[i].plot(x, central_difference(func, h, x))
        axs[i].set_title(f"{h=}")
        axs[i].plot(x, func_derivative(x), label="Analytical derivative")
        axs[i].legend()
    plt.show()

    # Assignment 1c
    # Use Ridder's method (Richardson extrapolation for differentiation)
    d = 2
    m = 5
    h = 1

    true_value = func_derivative(x)
    dn_dx = np.zeros(x.size)
    # dn_dx_generated = np.zeros(x.size)
    difference_between_analytical_and_calculated = np.zeros(x.size)
    for j, xvalue in enumerate(x):
        dn_dx[j], _, _ = ridders(func, h=h, x=xvalue, d=d, m=20)
        # dn_dx_generated[j], _, _ = ridders_derivative_generated(func, x=xvalue, h0=h, max_orders=20)
        difference_between_analytical_and_calculated[j] = abs(true_value[j] - dn_dx[j])
        if difference_between_analytical_and_calculated[j] > 1e-5:
            print(f"Warning: large difference between analytical and calculated derivative at x={xvalue}: {difference_between_analytical_and_calculated[j]}")
            print(j, xvalue, true_value[j], dn_dx[j], difference_between_analytical_and_calculated[j])
    plt.figure()
    plt.title(f"Ridder's method with initial h value {h}")
    plt.plot(x, dn_dx, label="Ridder's method")
    # plt.plot(x, dn_dx_test, label="Ridders' method test")
    plt.plot(x, true_value, label="Analytical derivative")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title(f"Difference between analytical and calculated derivative with initial h value {h}")
    plt.plot(x, np.log(difference_between_analytical_and_calculated))
    plt.show()

    # We can see slight errors around 0.8, 3 and 5.7.
    # The solutions also have three spikes with errors albeit at slightly different abscissa values.
    # Their reasoning is as follows:
    # If the initial h is too small, the first column of Di,j will be dominated by round-off error,
    # propagating numerical noise to the result and breaking stopping conditions.

    # fig, axs = plt.subplots(1, m)
    # plt.suptitle("Ridders method for numerical differentiation")
    # for i, derivative in enumerate(dn_dx):
    #     dn_dx = np.zeros(x.size)
    #         for j, xvalue in enumerate(x):
    #         dn_dx[j] = ridder(func, h=0.1, x=xvalue, d=d, m=m)
    #         axs[i].plot(x, d)
    #         axs[i].set_title(f"{h=}")


main()
