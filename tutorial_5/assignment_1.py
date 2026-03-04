from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np


def func(x):
    return x**2 * np.sin(x)


def func_derivative(x):
    return 2 * x * np.sin(x) + x**2 * np.cos(x)


def central_difference(func: Callable, h: float, x: np.ndarray):
    return func(x + h) - func(x - h) / (2 * h)


def ridder(func: Callable, h: float, x: np.ndarray, d: float = 2, m: int = 5):
    """Use Ridder's method (Richardson extrapolation for differentiation) to compute the numerical derivative of a function func,
    using the central differences formula with an initial h.
    d is the factor to decrease h with for successive approximations.
    m is the number of initial approximations.
    """
    D = np.zeros((m, len(x)))  # matrix storing m solution approximations for each x value in the input x array
    D[0] = central_difference(func, h, x)
    d_inv = 1 / d  # calculate this once for efficiency
    h_vector = np.zeros(m)
    h_vector[0] = h  # for plotting purposes, remove when we want only efficiency.
    for i in range(1, m):
        h *= d_inv
        h_vector[i] = h
        D[i] = central_difference(func, h, x)
    return D, h_vector


def main():

    # Assignment 1a
    x = np.linspace(0, 2 * np.pi, num=200)
    plt.figure()
    plt.plot(x, func_derivative(x))
    plt.title("Analytical derivative")

    # Assignment 1b
    # Use the central difference method to calculate the numerical derivative for different values of the step size h
    # It turns out the one with the largest step size (h=1) resembles the analytical derivative the best
    fig, axs = plt.subplots(1, 4)
    plt.suptitle("Central difference method for numerical differentiation")
    for i, h in enumerate([1, 0.1, 0.01, 0.001]):
        axs[i].plot(x, central_difference(func, h, x))
        axs[i].set_title(f"{h=}")
    # plt.show()

    # Assignment 1c
    # Use Ridder's method (Richardson extrapolation for differentiation)

    d = 2
    m = 5
    D, h_vector = ridder(func, h=0.1, x=x, d=d, m=m)
    fig, axs = plt.subplots(1, m)
    plt.suptitle("Ridders method for numerical differentiation")
    for i, (d, h) in enumerate(zip(D, h_vector)):
        axs[i].plot(x, d)
        axs[i].set_title(f"{h=}")
    plt.show()


main()
