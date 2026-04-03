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
    def __init__(self, x: np.ndarray, y: np.ndarray, partial_derivative_list: list[Callable], sigma, f: Callable, p: np.ndarray):
        """Implement the Levenberg Marquardt algorithm.
        x are the abscissa points of the measured data, y are the measured data values.
        partial_derivative_list should contain a list of the partial derivatives of the function to be fitted, with the partial derivative to each parameter.
        f is the function (model) to be fitted to the data, for which we try to find the best-fit parameters
        p is the initial estimation of the parameter vector.
        """
        self.x = x
        self.y_measured = y
        self.sigma: Union[float, list, tuple, np.ndarray] = sigma  # noqa: UP007 use old typing for compatibility with vdesk
        if np.any(self.sigma == 0):  # works for ints, floats, 1D and 2D arrays
            raise ValueError("No value in sigma can be zero!")
        self.f = f
        self.p = p
        self._construct_covariance_matrix()

    def _calculate_model(self):
        self.y_model = self.f(self.x, *self.p)  # unpack the vector of parameters into the function

    def _construct_jacobian(self, partial_derivative_list: list[Callable]):
        """Construct the Jacobian, the matrix holding partial derivatives i.e. dau y_i/dau p_j for as set of datapoints and another set of parameters.
        So why do we supply an x array instead of a y array? because we fill in x in the analytical partial derivative of each parameter
        """
        self.J = np.zeros((len(self.x), len(partial_derivative_list)))
        for i, x_value in enumerate(self.x):
            for j, partial_derivative in enumerate(partial_derivative_list):
                self.J[i, j] = partial_derivative(x_value)

    def _construct_covariance_matrix(self):
        number_of_data_points = len(self.y_measured)
        if isinstance(
            self.sigma, (int, float)
        ):  # This is the case for Least Squares, where the standard deviation is constant for all x_i, and there is no correlation between the sigmas for various x_i's
            self.std2_inv = 1 / self.sigma**2  # add to self for use in iterative calculation of chi square
            self.std2_inverse_matrix = np.identity(number_of_data_points) * self.std2_inv

        elif isinstance(
            self.sigma, (list, tuple)
        ):  # This is the case for Chi Squared, where the standard deviation can be different for each x_i, but there is no correlation between the sigmas for various x_i's
            if len(self.sigma) != number_of_data_points:
                raise ValueError("Length of standard deviation iterable should match number of datapoints")
            cov = np.identity(number_of_data_points)
            self.std2_inv_list = 1 / np.array(self.sigma) ** 2
            for i in range(len(cov)):
                cov[i, i] *= self.std2_inv_list[i]
            self.std2_inverse_matrix = cov
        elif isinstance(
            self.sigma, np.ndarray
        ):  # This is the most general case where the standard deviation can be different for each x_i, AND there is can be correlation between the sigmas for various x_i's
            if len(self.sigma) != number_of_data_points or self.sigma.shape[0] != self.sigma.shape[1]:
                raise ValueError("Length of standard deviation array should match number of datapoints and sigma should be a square array")
            self.std2_inverse_matrix = 1 / self.sigma**2
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
        self.alpha_accent = self.alpha + np.eye(self.alpha.shape[0]) * self.lmbda

    def _calculate_beta(self, y: np.ndarray):
        # Do measured y value minus predicted model y value!
        if len(y) != self.J.shape[0]:
            raise ValueError("y should be the same length as the number of rows of the Jacobian J!")
        self.beta = self.J.T @ self.std2_inverse_matrix @ y.T

    def _solve_delta_p(self):
        """Solve the linear system alpha delta_p = beta, for delta_p"""
        return np.linalg.solve(self.alpha_accent, self.beta)

    def _calculate_chisquare(self):
        """We calculate the chi squared every time we make a new estimation of our parameter vector p.
        The first will be a guess, the rest will follow from iteratively fitting our model to our data.
        The fit results from solving the linear system alpha delta_p = b for delta_p, where delta_p is the change in the parameter vector
        """
        if not isinstance(self.sigma, (list, tuple)):
            raise TypeError("sigma should be a list or tuple of standard deviations for use in chi square")
        # calculate the expected y values by putting the x values together with the current parameter estimation into the function
        return np.sum((self.y_measured - self.y_model) ** 2 * self.std2_inverse_matrix)

    def iteratively_improve_solution(self, weight: float = 10.0, improvement_threshold: float = 0.01):
        self._calculate_model()  # Calculate the y model values for the inicial parameter estimation p_0
        self.old_chisquare = self._calculate_chisquare()  # Step 1: calculate chisquare for p_0
        self.lmbda = 1e-3
        self._construct_jacobian(partial_derivative_list)
        self._calculate_alpha()
        self._calculate_beta(self.y_measured)
        delta_p = self._solve_delta_p()  # solve the linear system of equations to find the change in our parameter estimation
        self.p += delta_p
        self._calculate_model()
        self.new_chisquare = self._calculate_chisquare()
        while np.abs(self.new_chisquare - self.old_chisquare) > improvement_threshold:  # while solution keeps improving, keep going
            if self.new_chisquare > self.old_chisquare:  # Old solution was better, keep old solution
                self.lmbda *= weight  # We are far from the minimum, make bigger steps (more steepest descent)
            else:  # New solution is better, update old solution to new solution
                self.old_chisquare = self.new_chisquare
                self.lmbda /= weight  # We are close to the minimum, make smaller steps (more Quasi Newton)
            self._construct_jacobian(partial_derivative_list)
            self._calculate_alpha()
            self._calculate_beta(self.y_measured)
            delta_p = self._solve_delta_p()  # solve the linear system of equations to find the change in our parameter estimation
            self.p += delta_p
            self._calculate_model()
            self.new_chisquare = self._calculate_chisquare()
        return self.p  # self.p now contains the parameters for the best fit!


if __name__ == "__main__":
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

    sigma_list = [sigma for _ in range(number_of_datapoints)]
    initial_p = np.array([2.1, 1.1, 2.1])

    lm = LevenbergMarquardt(x, realizations[0], partial_derivative_list, sigma_list, func, initial_p)
    print(lm.iteratively_improve_solution())
