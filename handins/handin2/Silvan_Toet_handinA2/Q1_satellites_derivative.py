##### Derivative block #####


from collections.abc import Callable
from copy import deepcopy

import numpy as np


def dn_dx(x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> float | np.ndarray:
    """Analytical derivative of number density provide

    Parameters
    ----------
    x : ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Derivative of number density of
        satellite galaxies at given radius x.
    """
    return A * Nsat * b**3 * (x / b) ** a * np.exp(-((x / b) ** c)) * (-3 + a - c * (x / b) ** c) / x**4


def central_difference(func: Callable, h: float, x: np.ndarray) -> float:
    return func(x + h) - func(x - h) / (2 * h)


def ridder(func: Callable, h: float, x: np.ndarray, d: float = 2, m: int = 5, target_error: float = (np.finfo(float).eps) ** 0.8):  # noqa: RET503
    """Use Ridder's method (Richardson extrapolation for differentiation) to compute the numerical derivative of a function func,
    using the central differences formula with an initial h.
    d is the factor to decrease h with for successive approximations.
    m is the number of initial approximations.
    This approach is based on slide 8 of lecture 5.
    We will continue improving our estimate until the best estimate is smaller than the target error.
    The default target error is the machine relative error to the power 0.8, based on slide 9 of lecture 5.
    """
    d_inv = 1 / d  # calculate this once for efficiency
    h_vector = np.zeros(10 * m)
    improvement = 2 * target_error  # make sure while loop starts
    best_approx = 2**64
    j = 0
    while True:
        j += 1
        for i in range(m - j):
            if j == 0:
                h_vector[i] = central_difference(func, h, x)
                h *= d_inv
            else:
                # Now combine pairs of approximations (step 3 of slide 8), but with update rule adapted for 1D version (overwriting of previous approximations)
                h_vector[i] = (d ** (2 * j) * h_vector[i + 1] - h_vector[i]) / (d ** (2 * j) - 1)
        improvement = abs(best_approx - h_vector[0])
        if improvement < target_error:  # Terminate when improvement over previous best approx is smaller than target error
            return best_approx
        # if improvement > previous_improvement:  # How can I measure if error grows? like this?
        #     return best_approx
        # previous_improvement = improvement


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
            # if improvement > previous_improvement:  # How can I measure if error grows? like this?
            #     return best_approx
            # previous_improvement = improvement
