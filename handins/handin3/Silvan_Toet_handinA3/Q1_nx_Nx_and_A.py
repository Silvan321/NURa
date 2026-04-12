# I have moved these functions to a separate file to prevent circular import errors
from collections.abc import Callable

import numpy as np


def n(x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> np.ndarray:
    """
    Number density profile of satellite galaxies

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
    ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    return A * Nsat * (x / b) ** (a - 3) * np.exp(-((x / b) ** c))


def f(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return (x / b) ** (a - 3) * np.exp(-((x / b) ** c))


def N(x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> Callable:
    """N(x) dx is the number of satellites in infinitesimal range [x, x+dx).
    It is related to n(x) dx, the number density profile according to N(x) dx = n(x) 4pi x**2 dx."""
    return 4 * np.pi * A * Nsat * x ** (a - 1) * (1 / b) ** (a - 3) * np.exp(-((x / b) ** c))


#### Fitting ####
def get_normalization_constant(a: float, b: float, c: float, Nsat: float) -> float:
    """
    Calculate the normalization constant A (which is a function of a,b,c) for the satellite number density profile.

    Parameters
    ----------
    a : float
        Small-scale slope.
    b : float
        Transition scale.
    c : float
        Steepness of exponential drop-off.
    Nsat : float
        Average number of satellites.

    Returns
    -------
    float
        Normalization constant A.
    """
    # TODO: implement the calculation of the normalization constant
    return 0.0  # replace by the correct value
