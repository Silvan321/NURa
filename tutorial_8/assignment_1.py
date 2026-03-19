# Tutorial 8 Fitting of data


from functools import partial

from matplotlib import pyplot as plt
import numpy as np


def func(x, a, b, c):
    return a / x + b * x + c


def partial_a(x):
    return 1 / x


def partial_b(x):
    return x


def partial_c(x):
    return 1


def construct_jacobian(x: np.ndarray, partial_derivative_list: list[func]):
    """Construct the Jacobian, the matrix holding partial derivatives i.e. dau y_i/dau p_j for as set of datapoints and another set of parameters.
    So why do we supply an x array instead of a y array? because we fill in x in the analytical partial derivative of each parameter"""
    J = np.zeros((len(x), len(partial_derivative_list)))
    for i, x_value in enumerate(x):
        for j, partial_derivative in enumerate(partial_derivative_list):
            J[i, j] = partial_derivative(x_value)
    return J


def calculate_alpha(J: np.ndarray):
    """Alpha is the square matrix with nrows = ncols = the number of parameters in the function that we are fitting.
    The nrows and ncols should be equal to the number of columns in the supplied Jacobian."""
    alpha = np.zeros(J.shape[1])  # The nrows and ncols of alpha should be equal to the number of columns in the supplied Jacobian.

    # We need to calculate the sum of dau y_i/ dau p_k times dau y_i/dau p_l for the sum of i to N-1
    # Now for efficiency: we transpose the Jacobian which allows us to do matrix multiplication for each k,l combination
    # Row from k times i matrix times column i times l matrix


def main():
    a = 2
    b = 1
    c = 2
    func_const_filled = partial(func, a=a, b=b, c=c)

    number_of_datapoints = 20
    x = np.linspace(0.5, 4, num=number_of_datapoints)

    realizations = np.zeros((1000, number_of_datapoints))  # We want 1000 realizations of f(x) with different noise values on top. every f(x) consists of 20 datapoints
    for i, realization in enumerate(realizations):
        y_truth = func_const_filled(x)
        noise = np.random.normal(scale=0.1, size=number_of_datapoints)  # scale set smaller initially
        realization = y_truth + noise
        realizations[i] = realization

    plt.figure()
    plt.plot(x, func_const_filled(x))
    plt.plot(x, realizations[0])
    plt.show()


main()
