# Tutorial 8 Fitting of data


from collections.abc import Callable
from functools import partial

from typing import Union

import numpy as np
from matplotlib import pyplot as plt


def func(x, a, b, c):
    return a / x + b * x + c


def partial_a(x):
    return 1 / x


def partial_b(x):
    return x


def partial_c(x):
    return 1


class LevenbergMarquardt:
    def __init__(self, x: np.ndarray, y: np.ndarray, partial_derivative_list: list[Callable], sigma):
        self._construct_jacobian(x, partial_derivative_list)
        self._construct_covariance_matrix(len(y), sigma)
        self._calculate_alpha()
        self._calculate_beta(y)

    def _construct_jacobian(self, x: np.ndarray, partial_derivative_list: list[Callable]):
        """Construct the Jacobian, the matrix holding partial derivatives i.e. dau y_i/dau p_j for as set of datapoints and another set of parameters.
        So why do we supply an x array instead of a y array? because we fill in x in the analytical partial derivative of each parameter
        """
        self.J = np.zeros((len(x), len(partial_derivative_list)))
        for i, x_value in enumerate(x):
            for j, partial_derivative in enumerate(partial_derivative_list):
                self.J[i, j] = partial_derivative(x_value)

    def _construct_covariance_matrix(self, number_of_data_points: int, sigma: Union[float, list, tuple, np.ndarray]):  # noqa: UP007
        if isinstance(
            sigma, (int, float)
        ):  # This is the case for Least Squares, where the standard deviation is constant for all x_i, and there is no correlation between the sigmas for various x_i's
            cov = np.identity(number_of_data_points)
            self.std2_inverse_matrix = np.identity(number_of_data_points) / sigma**2
        elif isinstance(
            sigma, (list, tuple)
        ):  # This is the case for Chi Squared, where the standard deviation can be different for each x_i, but there is no correlation between the sigmas for various x_i's
            if len(sigma) != number_of_data_points:
                raise ValueError("Length of standard deviation iterable should match number of datapoints")
            cov = np.identity(number_of_data_points)
            for i in range(len(cov)):
                cov[i, i] /= sigma[i] ** 2
            self.std2_inverse_matrix = cov
        elif isinstance(
            sigma, np.ndarray
        ):  # This is the most general case where the standard deviation can be different for each x_i, AND there is can be correlation between the sigmas for various x_i's
            if len(sigma) != number_of_data_points or sigma.shape[0] != sigma.shape[1]:
                raise ValueError("Length of standard deviation array should match number of datapoints and sigma should be a square array")
            self.std2_inverse_matrix = 1 / sigma**2
        else:
            raise TypeError("Unknown type supplied to construct covariance matrix method")

    def _calculate_alpha(self):
        """Alpha is the square matrix with nrows = ncols = the number of parameters in the function that we are fitting.
        The nrows and ncols should be equal to the number of columns in the supplied Jacobian.
        std2_inverse_matrix is the inverse of the standard deviations squared of the elements.
        This forms the covariane matrix of the parameters.
        """
        # We need to calculate the sum of dau y_i/ dau p_k times dau y_i/dau p_l for the sum of i to N-1, for each combination of k and l.
        # Now for efficiency: we transpose the Jacobian which allows us to do matrix multiplication for each k,l combination
        # Row from k times i matrix times column i times l matrix
        self.alpha = self.J.T @ self.std2_inverse_matrix @ self.J

    def _calculate_beta(self, y: np.ndarray):
        # Do measured y value minus predicted model y value!
        if len(y) != self.J.shape[0]:
            raise ValueError("y should be the same length as the number of rows of the Jacobian J!")
        self.beta = self.J.T @ self.std2_inverse_matrix @ y.T

    def solve(self):
        """Solve the linear system alpha delta_p = beta, for delta_p"""
        return np.linalg.solve(self.alpha, self.beta)


def main():
    a = 2
    b = 1
    c = 2
    func_const_filled = partial(func, a=a, b=b, c=c)
    partial_derivative_list = [partial_a, partial_b, partial_c]
    sigma = 0.1  # scale set smaller initially

    number_of_datapoints = 20
    x = np.linspace(0.5, 4, num=number_of_datapoints)

    realizations = np.zeros((1000, number_of_datapoints))  # We want 1000 realizations of f(x) with different noise values on top. every f(x) consists of 20 datapoints
    for i, _ in enumerate(realizations):
        y_truth = func_const_filled(x)
        noise = np.random.normal(scale=sigma, size=number_of_datapoints)
        realization = y_truth + noise
        realizations[i] = realization

    plt.figure()
    plt.plot(x, func_const_filled(x))
    plt.plot(x, realizations[0])
    plt.show()

    lm = LevenbergMarquardt(x, realizations[0], partial_derivative_list, sigma)
    print(lm.solve())


main()
