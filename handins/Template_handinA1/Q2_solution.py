import os
import sys
import timeit

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["font.size"] = 20
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20


def load_data():
    """Function to load the data from Vandermonde.txt.

    Returns
    -------
    x (np.ndarray): Array of x data points.

    y (np.ndarray): Array of y data points.
    """
    data = np.genfromtxt(os.path.join(sys.path[0], "Vandermonde.txt"), comments="#", dtype=np.float64)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def construct_vandermonde_matrix(x: np.ndarray) -> np.ndarray:
    """Construct the Vandermonde matrix V with V[i,j] = x[i]^j.

    Parameters
    ----------
    x : np.ndarray, x-values.

    Returns
    -------
    V : np.ndarray, Vandermonde matrix.
    """
    V = np.zeros((len(x), len(x)), dtype=np.float64)
    for i, xvalue in enumerate(x):
        for j in range(len(x)):
            V[i, j] = xvalue**j
    return V


class LUDecomposition:
    def __init__(self, a: np.ndarray):
        """This class performs LU decomposition, where a matrix A is decomposed in two matrices L and U.
        Here L is a Lower Triangular matrix (only elements on or below diagonal), and U is an Upper triangular matrix (only elements on or above the diagonal).
        We can use these to solve the equation A*x = (L*U)*x = L*(U*x) = b, by first solving L*y=b and then U*x=y.
        Use the constructor of the class to do the decomposition, using Crout's algorithm (slide 12 lecture 3)
        Then the instance of the object holds the LU decomposed matrix.
        We can then call the solve method as many times as we like, for as many b's as we like.
        """
        self.nrows, self.ncolumns = a.shape[0], a.shape[1]
        self.LU_matrix = np.identity(self.ncolumns)  # step 1: construct an identity matrix to start
        for j in range(self.ncolumns):
            self.LU_matrix[0, j] = a[0, j]  # for i = 0, beta_0j = a_0j
            for i in range(1, self.nrows):  # starting from 0 or 1 should not make a difference, as the first row is already set to the correct values, but we can skip it to save some time
                if i <= j:
                    self.LU_matrix[i, j] = a[i, j] - sum(
                        self.LU_matrix[i, k] * self.LU_matrix[k, j] for k in range(i)
                    )  # beta_ij = a_ij - sum (alpha_ik beta_kj) from k=0 to i-1. range doesn't include end point!
                if i > j:
                    self.LU_matrix[i, j] = (a[i, j] - sum(self.LU_matrix[i, k] * self.LU_matrix[k, j] for k in range(j))) / self.LU_matrix[
                        j, j
                    ]  # alpha_ij = a_ij - sum (alpha_ik beta_kj) from k=0 to j-1. range doesn't include end point!

    def get_LU_decomposition(self):
        return self.LU_matrix

    def solve(self, b: np.ndarray) -> np.ndarray:
        # Forward substitution first: L*y=b
        b_size = len(b)
        y = np.zeros(b_size)
        x = np.zeros(b_size)
        y[0] = b[0]  # don't need to divide by LU_matrix[i,i] as for the forward substitution, the diagonal elements of L are all 1's
        for i in range(1, b_size):
            y[i] = b[i] - sum(self.LU_matrix[i, j] * y[j] for j in range(i))

        # Now do the backsubstitution
        x[b_size - 1] = y[b_size - 1] / self.LU_matrix[b_size - 1, b_size - 1]
        for i in reversed(range(b_size)):
            x[i] = (y[i] - sum(self.LU_matrix[i, j] * x[j] for j in range(i + 1, b_size))) / self.LU_matrix[i, i]
        return x


def evaluate_polynomial(c: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Evaluate y(x) = sum_j c[j] * x^j.

    Parameters
    ----------
    c : np.ndarray
        Polynomial coefficients.
    x_eval : np.ndarray
        Evaluation points.

    Returns
    -------
    y_eval : np.ndarray
        Polynomial values.
    """
    # evaluate the polynomial at x_eval using the coefficients c from vandermonde_solve_coefficients
    # evaluate efficiently using the formula on page 202 of the book "Numerical Recipes"
    polynomial_at_evaluated_points = np.zeros_like(x_eval)
    for i, x in enumerate(x_eval):
        p = c[-1]
        for j in reversed(range(len(c) - 1)):
            p = p * x + c[j]
        polynomial_at_evaluated_points[i] = p
    return polynomial_at_evaluated_points


def neville(x_data: np.ndarray, y_data: np.ndarray, x_interp: float) -> float:
    """
    Function that applies Nevilles algorithm to calculate the function value at x_interp.

    Parameters
    ------------
    x_data (np.ndarray): Array of x data points.
    y_data (np.ndarray): Array of y data points.
    x_interp (float): The x value at which to interpolate.

    Returns
    ------------
    float: The interpolated y value at x_interp.
    """
    # TODO:
    # write your Neville's algorithm
    return 0


# you can merge the function below with LU_decomposition to make it more efficient
def run_LU_iterations(
    x: np.ndarray,
    y: np.ndarray,
    iterations: int = 11,
    coeffs_output_path: str = "Coefficients_per_iteration.txt",
):
    """
    Iteratively improves computation of coefficients c.

    Parameters
    ----------
    x : np.ndarray
        x-values.
    y : np.ndarray
        y-values.
    iterations : int
        Number of iterations.
    coeffs_output_path : str
        File to write coefficient values per iteration.

    Returns
    -------
    coeffs_history :
        List of coefficient vectors.
    """
    # TODO:
    # Implement an iterative improvement for computing the coefficients c,
    # and save the coefficients at each iteration to coeffs_output_path.
    return [np.zeros_like(x) for _ in range(iterations)]  # Replace with your solution


def plot_part_a(
    x_data: np.ndarray,
    y_data: np.ndarray,
    coeffs_c: np.ndarray,
    plots_dir: str = "Plots",
) -> None:
    """
    Ploting routine for part (a) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    coeffs_c : np.ndarray
        Polynomial coefficients c.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """
    xx = np.linspace(x_data[0], x_data[-1], 1001)
    yy = evaluate_polynomial(coeffs_c, xx)
    y_at_data = evaluate_polynomial(coeffs_c, x_data)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0)
    axs[0].plot(xx, yy, linewidth=3)
    axs[0].set_xlim(np.floor(xx[0]) - 0.01 * (xx[-1] - xx[0]), np.ceil(xx[-1]) + 0.01 * (xx[-1] - xx[0]))
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(["data", "Via LU decomposition"], frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")
    axs[1].plot(x_data, np.abs(y_data - y_at_data), linewidth=3)

    plt.savefig(os.path.join(plots_dir, "vandermonde_sol_2a.pdf"))
    plt.close()


def plot_part_b(
    x_data: np.ndarray,
    y_data: np.ndarray,
    plots_dir: str = "Plots",
) -> None:
    """
    Ploting routine for part (b) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """
    xx = np.linspace(x_data[0], x_data[-1], 1001)
    yy = np.array([neville(x_data, y_data, x) for x in xx], dtype=np.float64)
    y_at_data = np.array([neville(x_data, y_data, x) for x in x_data], dtype=np.float64)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0)
    axs[0].plot(xx, yy, linestyle="dashed", linewidth=3)
    axs[0].set_xlim(np.floor(xx[0]) - 0.01 * (xx[-1] - xx[0]), np.ceil(xx[-1]) + 0.01 * (xx[-1] - xx[0]))
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(["data", "Via Neville's algorithm"], frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")
    axs[1].plot(x_data, np.abs(y_data - y_at_data), linestyle="dashed", linewidth=3)

    plt.savefig(os.path.join(plots_dir, "vandermonde_sol_2b.pdf"))
    plt.close()


def plot_part_c(
    x_data: np.ndarray,
    y_data: np.ndarray,
    coeffs_history: list[np.ndarray],
    iterations_num: list[int] = [0, 1, 10],
    plots_dir: str = "Plots",
) -> None:
    """
    Ploting routine for part (c) results.

    Parameters
    ----------
    x_data : np.ndarray
        x-values.
    y_data : np.ndarray
        y-values.
    coeffs_history : list[np.ndarray]
        Coefficients per iteration.
    iterations_num : list[int]
        Iteration numbers to plot.
    plots_dir : str
        Directory to save plots.

    Returns
    -------
    None
    """

    linstyl = ["solid", "dashed", "dotted"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    xx = np.linspace(x_data[0], x_data[-1], 1001)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0, 1.0])
    axs = gs.subplots(sharex=True, sharey=False)

    axs[0].plot(x_data, y_data, marker="o", linewidth=0, color="black", label="data")

    for i, k in enumerate(iterations_num):
        if k >= len(coeffs_history):
            continue
        c = coeffs_history[k]
        yy = evaluate_polynomial(c, xx)
        y_at_data = evaluate_polynomial(c, x_data)
        diff = np.abs(y_at_data - y_data)

        axs[0].plot(
            xx,
            yy,
            linestyle=linstyl[i],
            color=colors[i],
            linewidth=3,
            label=f"Iteration {k}",
        )
        axs[1].plot(x_data, diff, linestyle=linstyl[i], color=colors[i], linewidth=3)

    axs[0].set_xlim(np.floor(xx[0]) - 0.01 * (xx[-1] - xx[0]), np.ceil(xx[-1]) + 0.01 * (xx[-1] - xx[0]))
    axs[0].set_ylim(-400, 400)
    axs[0].set_ylabel("$y$")
    axs[0].legend(frameon=False, loc="lower left")

    axs[1].set_ylim(1e-16, 1e1)
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$|y - y_i|$")
    axs[1].set_xlabel("$x$")

    plt.savefig(os.path.join(plots_dir, "vandermonde_sol_2c.pdf"))
    plt.close()


def main():
    os.makedirs("Plots", exist_ok=True)
    x_data, y_data = load_data()
    V = construct_vandermonde_matrix(x_data)

    # I prefer a class implementation for the LU decomposition, based on the book "Numerical Recipes"
    # The constructor does the decomposition, the getter allows access to LU matrix itself
    ludcmp_instance = LUDecomposition(V)
    V_lu = ludcmp_instance.get_LU_decomposition()

    ludcmp_instance.solve(y_data)
    # compute times
    number = 10

    t_a = (
        timeit.timeit(
            stmt=lambda: ludcmp_instance.solve(y_data),
            number=number,
        )
        / number
    )

    xx = np.linspace(x_data[0], x_data[-1], 1001)
    t_b = (
        timeit.timeit(
            stmt=lambda: np.array([neville(x_data, y_data, x) for x in xx], dtype=np.float64),
            number=number,
        )
        / number
    )

    t_c = (
        timeit.timeit(
            stmt=lambda: run_LU_iterations(x_data, y_data, iterations=11),
            number=number,
        )
        / number
    )

    # write all timing
    with open("Execution_times.txt", "w", encoding="utf-8") as f:
        f.write(f"\\item Execution time for part (a): {t_a:.5f} seconds\n")
        f.write(f"\\item Execution time for part (b): {t_b:.5f} seconds\n")
        f.write(f"\\item Execution time for part (c): {t_c:.5f} seconds\n")
    c_a = ludcmp_instance.solve(y_data)
    plot_part_a(x_data, y_data, c_a)

    formatted_c = [f"{coef:.3e}" for coef in c_a]
    with open("Coefficients_output.txt", "w", encoding="utf-8") as f:
        for i, coef in enumerate(formatted_c):
            f.write(f"c$_{i + 1}$ = {coef}, ")

    plot_part_b(x_data, y_data)

    coeffs_history = run_LU_iterations(
        x_data,
        y_data,
        iterations=11,
        coeffs_output_path="Coefficients_per_iteration.txt",
    )
    plot_part_c(x_data, y_data, coeffs_history, iterations_num=[0, 1, 10])


if __name__ == "__main__":
    main()
