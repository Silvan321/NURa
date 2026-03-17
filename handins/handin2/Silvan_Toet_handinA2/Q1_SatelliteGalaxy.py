# imports
import sys
from collections.abc import Callable
from functools import partial
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Q1_satellites_derivative import dn_dx
from Q1_satellites_integrator import romberg_vector_version


def n(x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> float | np.ndarray:
    """Number density profile of satellite galaxies

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

    specific_integrand = partial(general_integrand, a=a, b=b, c=c)  # Use a partial function to set a, b and c to the given values in the general integrand, but still keep it as a function
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
