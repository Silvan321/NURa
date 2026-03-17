# imports
import sys
from collections.abc import Callable
from functools import partial
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Q1_satellites_derivative import central_difference, dn_dx
from Q1_satellites_integrator import romberg_vector_version
from Q1_satellites_sampling import additive_combined_rng, lcg, rng_64bit_xor_shift, sampler
from Q1_satellites_selection import choice, quicksort, selection_sort


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
    return A * Nsat * (x / b) ** (a - 3) * np.exp(-((x / b) ** c))


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

    rng_test_size = 100000
    P_64 = rng_64bit_xor_shift(size=rng_test_size, scale_uniform=True)
    P_lcg = lcg(size=rng_test_size, scale_uniform=True)
    P_add = additive_combined_rng(size=rng_test_size, scale_uniform=True)
    rngs_dict = {"64bit_XOR": P_64, "LCG": P_lcg, "Additive Combination": P_add}

    fig, axs = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f"Division of generated random numbers of the two sub generators and the combined generator,\nfor {rng_test_size} random numbers, scaled uniformly over 10 bins")
    for i, key in enumerate(rngs_dict):
        axs[i].set_title(key)
        axs[i].hist(rngs_dict[key])
    plt.savefig("Plots/rng_test.png", dpi=600)

    # Numerically determine maximum to normalize p(x) for sampling
    # Since this assigment doesn't cover material from lecture 7 maximization I don't use the methods described there
    x_max = max(np.linspace(xmin, xmax, N_generate), key=lambda x: n(x, A, Nsat, a, b, c))
    pmax = n(x_max, A, Nsat, a, b, c)

    p_of_x_norm = lambda x: n(x, A, Nsat, a, b, c) / pmax
    random_samples = sampler(p_of_x_norm, xmin, xmax, N_generate)

    edges = 10 ** np.linspace(np.log10(xmin), np.log10(xmax), 21)

    # Note that we use the density parameter to divide each bin by its width, which is what we are asked to do. This parameter automatically causes the integral/sum of all the bins to add up to 1
    # So even though we generate 10000 samples, we have to MULTIPLY WITH 100 instead of divide with 100 to correctly scale the histogram, since the average total number of satellites over our range of interest is 100.
    hist = np.histogram(xmin + random_samples * (xmax - xmin), bins=edges, density=True)[0]
    hist_scaled = 1e2 * hist

    relative_radius = edges.copy()
    analytical_function = n(relative_radius, A, Nsat, a, b, c)

    fig1b, ax = plt.subplots()
    ax.stairs(hist_scaled, edges=edges, fill=True, label="Satellite galaxies")  # just an example line, correct this!
    plt.plot(relative_radius, analytical_function, "r-", label="Analytical solution")  # correct theiis according to the exercise!
    ax.set(
        xlim=(xmin, xmax),
        ylim=(10 ** (-2), 1000),  # you may or may not need to change ylim
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.savefig("Plots/my_solution_1b.png", dpi=600)

    # # Cumulative plot of the chosen galaxies (1c)
    chosen_indices = choice(N_generate, Nsat)  # Our choice function generates 100 random indices from the N_generate = 10000 possible options, without rejection or duplication using a full period LCG
    unique_indices = np.unique(chosen_indices)  # I understood from Marcel that for testing our RNG (in this case for generating every value exactly once), it is okay to use built-in functions
    plt.figure()
    plt.title(f"Histogram of chosen indices for {Nsat} points.\n The number of unique indices is {unique_indices.size}")
    plt.hist(chosen_indices)
    plt.savefig("Plots/choice_test.png", dpi=600)
    chosen = selection_sort(random_samples[chosen_indices])  #  100 selected and sorted random numbers in the range (0,10000) that act as indices for the larger generated sample of 10000 galaxies

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
    dn_dx_numeric = central_difference(func=func_to_eval, h=0.1, x=x_to_eval)  # I unfortunately could not get Ridder's method to work so I use the central difference method
    dn_dx_analytic = dn_dx(x_to_eval, A, Nsat, a, b, c)
    with open("Calculations/satellite_deriv_analytic.txt", "w") as f:
        f.write(f"{dn_dx_analytic:.12g}\n")

    with open("Calculations/satellite_deriv_numeric.txt", "w") as f:
        f.write(f"{dn_dx_numeric:.12g}\n")


if __name__ == "__main__":
    main()
