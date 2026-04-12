##### Integrator block #####

import numpy as np


def trapezoid(a, b, func, N: int):
    """Use the extended trapezoid rule to calculate the integral.
    Parameters a and b are start and stop x values of the range to be integrated respectively.
    func is the function to be evaluated.
    N is the number of evaluations.
    """
    xdata = np.linspace(a, b, num=N)
    h = (b - a) / N  # step size
    return h * (0.5 * (func(b) + func(a)) + np.sum(func(xdata[1:-1])))


def romberg_vector_version(a, b, func, N_start: int = 100, order: int = 5, return_error: bool = False):
    """See slide 14 of Lecture 4 annotated slides for the formula used.
    N_start is the initial number of function evaluations. This will double with every order.
    order is the number of initial approximations.
    We return the best estimate for the integral stored at r_0
    If return_error is True, the error estimate abs(r_0 - r_1) is also returned
    """
    romberg_vector = np.zeros(shape=order)
    for j in range(order):
        for i in range(order - j):
            if j == 0:  # the first time we fill the column, we use the trapezoid rule with doubling step sizes per row entry
                N = (2**i) * N_start  # number of intervals doubles with each depth
                romberg_vector[i] = trapezoid(a, b, func, N)
            else:  # Subsequent times we use the update rule
                romberg_vector[i] = (4**j * romberg_vector[i + 1] - romberg_vector[i]) / (4**j - 1)  # General version of 4/3 S_1 - 1/3 S_0
    if return_error:
        return romberg_vector[0], abs(romberg_vector[0] - romberg_vector[1])  # (value, error)
    return romberg_vector[0]
