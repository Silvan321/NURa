# imports
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Q1_golden_section_minimizer import bracket_minimum, golden_section_search
from Q1_Levenberg_Marquardt_minimizer import LevenbergMarquardt, partial_derivative_list
from Q1_nx_Nx_and_A import N_func, get_normalization_constant, n_func
from Q1_poisson import negative_poisson_ln_likelihood

this_directory = Path(__file__).resolve().parents[0]  # Use this directory for absolute imports to allow running the script in debug mode
sys.path.append(this_directory)  # Append directory to sys path to fix relative import shenanigans


def readfile(filename):
    """
    Helper function to read in the satellite galaxy data from the provided text files.

    Parameters
    ----------
    filename : str
        The name of the file to read in.

    Returns
    -------
    radius : ndarray
        The virial radius for all the satellites in the file.
    nhalo : int
        The number of halos in the file.
    """
    f = open(filename, "r")
    data = f.readlines()[3:]  # Skip first 3 lines
    nhalo = int(data[0])  # number of halos
    radius = []

    for line in data[1:]:
        if line[:-1] != "#":
            radius.append(float(line.split()[0]))

    radius = np.array(radius, dtype=float)
    f.close()
    return (
        radius,
        nhalo,
    )  # Return the virial radius for all the satellites in the file, and the number of halos


def bin_data(x_all, n_haloes, n_bins, x_min, x_max, log_bins):
    """
    Returns:
        bin_centers
        N_i  (mean observed satellites per halo per bin)
        edges
    """

    if log_bins:
        edges = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
    else:
        edges = np.linspace(0.0, x_max, n_bins + 1)

    counts, _ = np.histogram(x_all, bins=edges)

    # Convert to mean per halo
    N_i = counts / n_haloes

    # Bin centers (geometric for log bins, based on lecture 9 slide 31: x_log = sqrt(x1*x2) -> log(x_log) = 0.5 * (x1 + x2))
    if log_bins:
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, N_i, edges


def minimize_poisson_ln_likelihood(model: callable, data: np.ndarray, initial_params: tuple) -> tuple:
    """
    Minimize the Poisson negative log-likelihood for a given model and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    initial_params : tuple
        Initial guess for the parameters to minimize over.

    Returns
    -------
    best_params : tuple
        The parameters that minimize the Poisson negative log-likelihood value.
    min_ln_likelihood : float
        The minimum Poisson negative log-likelihood value achieved.
    """

    # TODO: implement the minimization of the Poisson negative log-likelihood using your custom method. Remember to normalize for each minimization step

    best_params = initial_params
    min_ln_likelihood = negative_poisson_ln_likelihood(model, data, initial_params)  # replace by the correct calculation of the Poisson negative log-likelihood for the given parameters

    return best_params, min_ln_likelihood


# =====================================================
# ======== Main functions for each subquestion ========
# =====================================================


def do_question_1a():
    # ======== Question 1a: Maximization of N(x) ========
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    A_1a = 256 / (5 * np.pi ** (3 / 2))
    bounds_for_maximization_and_integration = (0, 5)
    x_lower, x_upper = 10**-4, 5
    x_range = np.linspace(x_lower, x_upper, num=1000)  # TODO: maybe replace by log axis

    fig1a, axs = plt.subplots(1, 2, figsize=(16, 10))
    plt.suptitle("n(x) dx, the number density profile, compared to N(x) dx, \nthe number of satellites in the infinitesimal range [x, x+dx)")
    axs[0].plot(x_range, n_func(x_range, A_1a, Nsat, a, b, c))
    axs[0].set_title("n(x) dx")
    axs[1].plot(x_range, N_func(x_range, A_1a, Nsat, a, b, c))
    axs[1].set_title("N(x) dx")
    plt.savefig("Plots/nx_vs_Nx.png", dpi=600)
    plt.close()

    # First we want to create a 3 point bracket which brackets our maximum
    # Since our algorithms are designed to find the minimum, we have to input the negative of the function in question

    N_1a = partial(N_func, A=A_1a, Nsat=Nsat, a=a, b=b, c=c)  # create a partial function with all variables that are given for Q1a fixed
    N_1a_negative = lambda x: -N_1a(x)  # Create the negative of the function
    three_point_bracket = bracket_minimum(func=N_1a_negative, a=1, b=1.1)
    print(three_point_bracket)
    x_max, number_of_iterations = golden_section_search(N_1a_negative, *three_point_bracket)
    Nx_max = N_1a(x_max)
    print(f"{x_max=}, {Nx_max=}, {number_of_iterations=}")
    # replace with calculation of the maximum of N(x) based on n(x, A, Nsat, a, b, c) and your minimizer
    # We will use our Golden Section Search minimizer to find the maximum of N(x).

    # Note that I am a bit confused that the template used 1e-4 to 5 as the x range to find the maximum (and did not say to replace the bounds), whereas the question asks to find the maximum between 0 and 5.
    # When I replaced the bounds 0 to 5 in the previous handin for the integration constant calculation to 1e-20 to 5 to solve a 0^x problem in Python, I lost points
    # I have now incorporated the hint that this 0^x problem can be alleviated by combining the x^2 with the x^(a-3) term so I will use the true 0 to 5 x range in my calculation for A this time,
    # Since I don't expect a to go below 1 while fitting the data and give the same problem.

    # Write the results to text files for later use in the PDF

    with open(this_directory / "Calculations/satellite_max_x.txt", "w") as f:
        f.write(f"{x_max:.6f}")
    with open(this_directory / "Calculations/satellite_max_Nx.txt", "w") as f:
        f.write(f"{Nx_max:.6f}")


def do_question_1b():
    # ======== Question 1b: Fitting N(x) with chi-squared ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    N_sat = []
    min_chi2_values = []
    best_params_chi2 = []

    # initialize figure with 5 subplots on 3x2 grid for the 5 data files
    fig, axs = plt.subplots(3, 2, figsize=(6.4, 8.0))
    axs = axs.flatten()

    for datafile in datafiles:
        radius, nhalo = readfile(this_directory / f"Data/satgals_{datafile}.txt")
        print(f"{datafile=}, {np.min(radius)=}, {np.max(radius)=} {nhalo=}")

        x_lower, x_upper = (10**-4, 5)
        bins = 30
        centers, N_i, edges = bin_data(radius, nhalo, n_bins=30, x_min=x_lower, x_max=x_upper, log_bins=True)

        print("Number of haloes:", nhalo)
        print("First few N_i:", N_i[:5])
        Nsat = len(radius) / nhalo  # Mean number of satellites in each halo should be the number of satellites divided by the number of halos
        print(f"{np.sum(N_i)=}")  # This should be equal to the sum of all mean number of satellites per halo for all radial bins
        print(f"{Nsat=}")

        # plt.step(centers, N_i, where="mid")
        # plt.xscale("log")
        # plt.xlabel("x = r / r_vir")
        # plt.ylabel("N_i (per halo)")
        # plt.title("Binned satellite profile")
        # plt.show()

        # TODO: implement the fitting of N(x) to the data using chi-squared minimization.
        sigma = 0.001  # scale set smaller initiall
        sigma_list = [sigma for _ in range(bins)]
        a_1a = 2.4
        b_1a = 0.25
        c_1a = 1.6
        A_1a = 256 / (5 * np.pi ** (3 / 2))
        initial_p = np.array([A_1a, Nsat, a_1a, b_1a, c_1a], dtype=np.float64)  # Use values from 1a as initial parameter estimates for A, a, b and c. Use observed Nsat as model Nsat

        lm = LevenbergMarquardt(centers, N_i, partial_derivative_list, sigma_list, N_func, initial_p, linear=True)
        print(lm.iteratively_improve_solution())

        # Store N_sat, chi2 values and best-fit parameters in their arrays
        N_sat.append(Nsat)
        min_chi2_values.append(0.0)
        best_params_chi2.append((0.0, 0.0, 0.0))  # replace by the correct best-fit parameters (a,b,c) found from chi-squared minimization

        # Plot the data and the best-fit model for each data file in a subplot.
        axs[datafiles.index(datafile)].hist([], bins=bins)  # plot the histogram of the data

        x_plot = np.linspace(x_lower, x_upper, 100)  # create x_array for plotting the model
        axs[datafiles.index(datafile)].plot(x_plot, np.ones_like(x_plot))  # plot the best-fit model using the best-fit parameters found from chi-squared minimization

        # Add labels and title to the subplot
        axs[datafiles.index(datafile)].set_title(f"Data file: {datafile}")
        axs[datafiles.index(datafile)].set_xlabel("x = r / r_virial")
        axs[datafiles.index(datafile)].set_ylabel("Number of satellites")

        # log-log scaling
        axs[datafiles.index(datafile)].set_xscale("log")
        axs[datafiles.index(datafile)].set_yscale("log")

    # Save the figure with all subplots
    plt.tight_layout()
    plt.savefig("Plots/satellite_fits_chi2.png")

    # Save N_sat, chi2 values and best-fit parameters for each data file to tex files for later use in the PDF
    with open("Calculations/table_fitparams_chi2.tex", "w") as f:
        rows = list(zip(N_sat, min_chi2_values, best_params_chi2))
        for idx, (N, chi2_val, params) in enumerate(rows):
            a, b, c = params
            line_end = " \\\\" if idx < len(rows) - 1 else ""
            f.write(f"m{idx + 11} & {N:.5f} & {chi2_val:.5f} & {a:.5f} & {b:.5f} & {c:.5f}{line_end}\n")


def do_question_1c():
    # ======== Question 1c: Fitting N(x) with Poisson ln-likelihood ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    min_poisson_llh_values = []
    best_params_poisson = []

    # initialize figure with 5 subplots on 3x2 grid for the 5 data files
    fig, axs = plt.subplots(3, 2, figsize=(6.4, 8.0))
    axs = axs.flatten()

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")
        x_lower, x_upper = (
            10**-4,
            5,
        )  # replace by appropriate limits for x based on the data

        # TODO: implement fit using Poisson negative log-likelihood minimization.

        # Store poisson llh values and best-fit parameters in their arrays
        min_poisson_llh_values.append(0.0)
        best_params_poisson.append((0.0, 0.0, 0.0))  # replace by the correct best-fit parameters (a,b,c) found from Poisson negative log-likelihood minimization

        # Plot the data and the best-fit model for each data file in a subplot.
        axs[datafiles.index(datafile)].hist([], bins=10)  # plot the histogram of the data
        x_plot = np.linspace(x_lower, x_upper, 100)  # create x_array for plotting the model
        axs[datafiles.index(datafile)].plot(x_plot, np.ones_like(x_plot))  # plot the best-fit model using the best-fit parameters found from Poisson negative log-likelihood minimization

        # Add labels and title to the subplot
        axs[datafiles.index(datafile)].set_title(f"Data file: {datafile}")
        axs[datafiles.index(datafile)].set_xlabel("x = r / r_virial")
        axs[datafiles.index(datafile)].set_ylabel("Number of satellites")

        # log-log scaling
        axs[datafiles.index(datafile)].set_xscale("log")
        axs[datafiles.index(datafile)].set_yscale("log")

    # Save the figure with all subplots
    plt.tight_layout()
    plt.savefig("Plots/satellite_fits_poisson.png")

    # Save poisson llh values and best-fit parameters for each data file to text files for later use in the PDF
    with open("Calculations/table_fitparams_poisson.tex", "w") as f:
        rows = list(zip(min_poisson_llh_values, best_params_poisson))
        for idx, (llh_val, params) in enumerate(rows):
            a, b, c = params
            line_end = " \\\\" if idx < len(rows) - 1 else ""
            f.write(f"m{idx + 11} & {llh_val:.5f} & {a:.5f} & {b:.5f} & {c:.5f}{line_end}\n")


def do_question_1d():
    # ======== Question 1d: Statistical tests ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    G_scores_chi2 = []
    Q_scores_chi2 = []

    G_scores_poisson = []
    Q_scores_poisson = []

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")

        # Use best-fit parameters from previous steps
        best_params_chi2 = (0.0, 0.0, 0.0)  # replace by the correct array
        best_params_poisson = (0.0, 0.0, 0.0)  # replace by the correct array

        # TODO: implement the statistical tests to calculate G and Q scores for both chi2 and poisson fits, and store the results in their respective arrays

        # Append the G and Q scores for chi2 and poisson fits to their respective arrays
        G_scores_chi2.append(0.0)
        Q_scores_chi2.append(0.0)
        G_scores_poisson.append(0.0)
        Q_scores_poisson.append(0.0)

    # Save G and Q scores for chi2 and poisson fits to tex files for later use in the PDF
    with open("Calculations/statistical_test_table_rows.tex", "w") as f:
        rows = []
        for i, (G, Q) in enumerate(zip(G_scores_chi2, Q_scores_chi2), start=11):
            rows.append(f"$\\chi^2$ & m{i} & {G:.5f} & {Q:.5f}")

        for i, (G, Q) in enumerate(zip(G_scores_poisson, Q_scores_poisson), start=11):
            rows.append(f"Poisson & m{i} & {G:.5f} & {Q:.5f}")

        for idx, row in enumerate(rows):
            if idx < len(rows) - 1:
                f.write(row + " \\\\\n")
            else:
                f.write(row)


def do_question_1e():
    # ======== Question 1e: Monte Carlo simulations ========
    # pick one of the data files to perform the Monte Carlo simulations on, e.g. m12
    datafiles = ["m11", "m12", "m13", "m14", "m15"]
    index = 1  # index of the data file to use for Monte Carlo simulations, e.g. 1 for m12

    radius, nhalo = readfile(f"Data/satgals_{datafiles[index]}.txt")

    # Use best-fit parameters from previous steps for the original data file
    best_params_chi2 = (0.0, 0.0, 0.0)  # replace by the correct array
    best_params_poisson = (0.0, 0.0, 0.0)  # replace by the correct array

    pseudo_chi2_params = []
    pseudo_poisson_params = []

    num_pseudo_experiments = 10  # replace by number with reasonable runtime
    for i in range(num_pseudo_experiments):
        # TODO: generate pseudo-data by sampling from original best-fit chi2 and poisson models
        # Then, for each pseudo-dataset, perform the chi2 and poisson fits to find the best-fit parameters.

        # Append the best-fit parameters for each pseudo-dataset to their respective arrays.
        pseudo_chi2_params.append((0.0, 0.0, 0.0))  # replace by the correct best-fit parameters (a,b,c) found from chi-squared minimization for the pseudo-dataset
        pseudo_poisson_params.append((0.0, 0.0, 0.0))  # replace by the correct best-fit parameters (a,b,c) found from Poisson negative log-likelihood minimization for the pseudo-dataset

    # plot the pseudo best-fit profiles, plot the original best-fit profile in another color and plot the mean in one more color

    # chi2 plot
    x_plot = np.linspace(1e-4, 5, 100)  # create x_array for plotting the model
    plt.figure(figsize=(6.4, 4.8))
    for params in pseudo_chi2_params:
        plt.plot(x_plot, np.ones_like(x_plot))  # plot the best-fit model for each pseudo-dataset using the best-fit parameters found from chi-squared minimization

    plt.plot(x_plot, np.ones_like(x_plot))  # plot the original best-fit model using the best-fit parameters found from chi-squared minimization on the real data

    mean_params_chi2 = np.mean(pseudo_chi2_params, axis=0)  # calculate the mean of the best-fit parameters from the pseudo-datasets
    plt.plot(x_plot, np.ones_like(x_plot))  # plot the mean of the best-fit models from the pseudo-datasets

    plt.title(f"Monte Carlo simulations - chi2 fit - Data file: {datafiles[index]}")
    plt.xlabel("x = r / r_virial")
    plt.ylabel("Number of satellites")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(["Pseudo-dataset fits", "Original fit", "Mean of pseudo fits"])
    plt.savefig("Plots/satellite_monte_carlo_chi2.png")

    # poisson plot
    x_plot = np.linspace(1e-4, 5, 100)  # create x_array for plotting the model
    plt.figure(figsize=(6.4, 4.8))
    for params in pseudo_poisson_params:
        plt.plot(x_plot, np.ones_like(x_plot))  # plot the best-fit model for each pseudo-dataset using the best-fit parameters found from Poisson negative log-likelihood minimization
    plt.plot(x_plot, np.ones_like(x_plot))  # plot the original best-fit model using the best-fit parameters found from Poisson negative log-likelihood minimization on the real data

    mean_params_poisson = np.mean(pseudo_poisson_params, axis=0)  # calculate the mean of the best-fit parameters from the pseudo-datasets
    plt.plot(x_plot, np.ones_like(x_plot))  # plot the mean of the best-fit models from the pseudo-datasets

    plt.title(f"Monte Carlo simulations - Poisson fit - Data file: {datafiles[index]}")
    plt.xlabel("x = r / r_virial")
    plt.ylabel("Number of satellites")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(["Pseudo-dataset fits", "Original fit", "Mean of pseudo fits"])
    plt.savefig("Plots/satellite_monte_carlo_poisson.png")


if __name__ == "__main__":
    do_question_1a()
    do_question_1b()
    do_question_1c()
    do_question_1d()
    do_question_1e()
