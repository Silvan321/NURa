# imports
from functools import partial
from math import pi
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


def n(x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> float | np.ndarray:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : float | ndarray
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
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    return 0  # insert your function


##### Integrator block #####


def trapezoid(a, b, func, N: int):
    """Use the extended trapezoid rule to calculate the integral.
    Parameters a and b are start and stop x values of the range to be integrated respectively.
    func is the function to be evaluated.
    N is the number of evaluations.
    """
    xdata = np.linspace(a, b, num=N)
    h = (b - a) / N  # step size
    return h * (0.5 * (func(b) + func(a)) + np.sum(func(xdata[1:-1])))


def romberg_vector_version(a, b, func, N_start: int, order: int = 5, return_error: bool = False):
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


#### Sampler block ####


def sampler(
    dist: callable,
    min: float,
    max: float,
    Nsamples: int,
    args: tuple = (),
) -> np.ndarray:
    """
    Sample a distribution using sampling method of your choice

    Parameters
    dist : callable
        Distribution to sample
    min :
        Minimum value for sampling
    max : float
        Maximum value for sampling
    Nsamples : int
        Number of samples
    args : tuple, optional
        Arguments of the distribution to sample, passed as args to dist

    Returns
    -------
    sample: ndarray
        Values sampled from dist, shape (Nsamples,)
    """

    return np.zeros(Nsamples)


#### Sorting block ####


def sort_array(
    arr: np.ndarray,
    inplace: bool = False,
) -> np.ndarray:
    """
    Sort a 1D array using a sorting algorithm of your choice

    Parameters
    ----------
    arr : ndarray
        Input array to be sorted
    inplace : bool, optional
        If True, sort the array in-place
        If False, return a sorted copy

    Returns
    -------
    sorted_arr : ndarray
        Sorted array (same shape as arr)

    """
    if inplace:
        sorted_arr = arr
    else:
        sorted_arr = arr.copy()

    # TODO: sort sorted_arr in-place here

    return sorted_arr


def choice(arr: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Choose given number of random elements from an array, without replacement

    Parameters
    ----------
    arr : ndarray
        Array to shuffle
    size : int, optional
        Number of elements to pick from array
        The default is 1

    Returns
    -------
    chosen : ndarray
        Randomly chosen elements from arr, shape (size,)
    """
    # TODO: Implement your choice function here, e.g. by using Fisher-Yates shuffling
    return arr[:size].copy()


##### Derivative block #####


def dn_dx(x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> float | np.ndarray:
    """
    Analytical derivative of number density provide

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
    # TODO: Write the analytical derivative of n(x) here
    return 0.0


def finite_difference(function: callable, x: float | np.ndarray, h: float) -> float | np.ndarray:
    """
    A building block to compute derivative using finite differences

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h : float
        Step size for finite difference

    Returns
    -------
    dy : float | ndarray
        Derivative at x
    """
    # TODO: Implement finite difference method
    return 0.0


def compute_derivative(
    function: callable,
    x: float | np.ndarray,
    h_init: float,
    # For Ridders use parameters below:
    # d: float, # Factor by which to decrease h_init every iteration
    # eps: float, # Relative error
    # max_iters: int = 10, 3 Maximum number of iterations before exiting
) -> float | np.ndarray:
    """
    Function to compute derivative

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h_init : float
        Initial step size for finite difference

    Returns
    -------
    df : float | ndarray
        Derivative at x
    """
    # TODO: Implement derivative
    return 0.0


def general_integrand(x: np.ndarray, a: float, b: float, c: float) -> Callable:
    return x**2 * (x / b) ** (a - 3) * np.exp(-((x / b) ** c))


def main():

    # Values from the hand-in
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    bounds = (1e-20, 5)  # Use 1e-20 for the lower bound of the integral, because Python complains if we raise 0 to a negative power (a-3 = 2.4-3 = -0.6)
    xmin, xmax = 10**-4, 5
    N_generate = 10000
    xx = np.linspace(xmin, xmax, N_generate)

    specific_integrand = partial(general_integrand, a=a, b=b, c=c)
    integral, err = romberg_vector_version(a=bounds[0], b=bounds[1], func=specific_integrand, N_start=100, order=5, return_error=True)

    # Normalisation
    A = 1 / (4 * pi * integral)  # to be computed
    with open("Calculations/satellite_A.txt", "w") as f:
        f.write(f"{A:.12g}\n")
    integrand = lambda x, a, b, c: 0.0  # replace by the correct function
    integrated_Nsat = 0.0  # replace by the correct integral, e.g. by calling your integrator

    p_of_x = lambda x: 0.0  # replace by the normalised distribution of satellite galaxies as a function of x

    # Numerically determine maximum to normalize p(x) for sampling
    pmax = 0.0  # replace by taking the maximum value of p_of_x

    p_of_x_norm = lambda x: 0.0  # replace by the normalised distribution
    random_samples = np.zeros(N_generate)  # replace by your sampler(p_of_x_norm, min=xmin, max=xmax, Nsamples=N_generate, args=())

    edges = 10 ** np.linspace(np.log10(xmin), np.log10(xmax), 21)

    hist = np.histogram(xmin + np.sort(np.random.rand(N_generate)) * (xmax - xmin), bins=edges)[0]  # replace!
    hist_scaled = 1e-3 * hist  # replace; this is NOT what you should be plotting, this is just a random example to get a plot with reasonable y values (think about how you *should* scale hist)

    fig = plt.figure()
    relative_radius = edges.copy()  # replace!
    analytical_function = edges.copy()  # replace

    fig1b, ax = plt.subplots()
    ax.stairs(hist_scaled, edges=edges, fill=True, label="Satellite galaxies")  # just an example line, correct this!
    plt.plot(relative_radius, analytical_function, "r-", label="Analytical solution")  # correct this according to the exercise!
    ax.set(
        xlim=(xmin, xmax),
        ylim=(10 ** (-3), 10),  # you may or may not need to change ylim
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.savefig("Plots/my_solution_1b.png", dpi=600)

    # Cumulative plot of the chosen galaxies (1c)
    chosen = xmin + np.sort(np.random.rand(Nsat)) * (xmax - xmin)  # replace!
    fig1c, ax = plt.subplots()
    ax.plot(chosen, np.arange(100))
    ax.set(
        xscale="log",
        xlabel="Relative radius",
        ylabel="Cumulative number of galaxies",
        xlim=(xmin, xmax),
        ylim=(0, 100),
    )
    plt.savefig("Plots/my_solution_1c.png", dpi=600)

    x_to_eval = 1
    func_to_eval = lambda x: n(x, A, Nsat, a, b, c)
    dn_dx_numeric = 0.0  # replace by your derivative, e.g. compute_derivative(func_to_eval, x_to_eval, h_init=0.1)
    dn_dx_analytic = dn_dx(x_to_eval, A, Nsat, a, b, c)
    with open("Calculations/satellite_deriv_analytic.txt", "w") as f:
        f.write(f"{dn_dx_analytic:.12g}\n")

    with open("Calculations/satellite_deriv_numeric.txt", "w") as f:
        f.write(f"{dn_dx_numeric:.12g}\n")


if __name__ == "__main__":
    main()
