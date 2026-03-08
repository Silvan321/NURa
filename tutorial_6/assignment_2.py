# Root finding

from collections.abc import Callable
from matplotlib import pyplot as plt
import numpy as np


def bisection(func: Callable, a: float, b: float, abs_err: float, rel_err: float, max_number_of_iterations: int = 50):
    """Implement the bisection algorithm for finding a single root for a 1D function.
    a and b are the boundaries of the interval in which the root should lie.
    """
    fa, fb = func(a), func(b)
    if not fa * fb < 0:  # if the root is inside the bracket, one function evaluation is positive and the other is negative, so the product must be negative!
        raise ValueError(
            "Product of function evaluations not negative. If the root is inside the bracket, one function evaluation is positive and the other is negative, so the product must be negative!"
        )

    fc = -1
    iteration = 0
    while not np.isclose(fc, 0.0, rtol=rel_err, atol=abs_err) and iteration < max_number_of_iterations:
        c = (a + b) / 2
        fc = func(c)  # middle function evaluation
        if fc * fa < 0:  # root is in between low and middle value
            fb = fc  # update function evaluation of high value to middle value
            b = c  # update bracket for next c calculation
        else:  # root is in between middle and high value
            fa = fc  # update function evaluation of low value to middle value
            a = c
        iteration += 1
    return c, iteration


def secant(func: Callable, a: float, b: float, abs_err: float, rel_err: float, max_number_of_iterations: int = 50):
    """Implement the secant algorithm for finding a single root for a 1D function.
    a and b are the boundaries of the interval in which the root should lie.
    """
    fa, fb = func(a), func(b)
    if not fa * fb < 0:  # if the root is inside the bracket, one function evaluation is positive and the other is negative, so the product must be negative!
        raise ValueError(
            "Product of function evaluations not negative. If the root is inside the bracket, one function evaluation is positive and the other is negative, so the product must be negative!"
        )

    fc = -1
    iteration = 0
    while not np.isclose(fc, 0.0, rtol=rel_err, atol=abs_err) and iteration < max_number_of_iterations:
        c = b - ((b - a) / (fb - fa)) * fb
        fc = func(c)  # middle function evaluation
        iteration += 1
        b, a = c, b  # update b and a (x_i and x_i-1) to c and b (x_i+1 and x_i) respectively
        fa, fb = func(a), func(b)
    return c, iteration


def func2a(x):
    return x**3 - 6 * x**2 + 11 * x - 6


def func2b(x):
    return np.tan(np.pi * x) - 6


def func2c(x):
    return x**3 - 2 * x + 2


def func2d(x):
    return np.exp(10 * (x - 1)) - 1 / 10


def main():

    # Know thy function!
    func_list = [func2a, func2b, func2c, func2d]
    algorithm_list = [bisection, secant]
    total_x = np.linspace(-2, 4, num=1000)

    fig, axs = plt.subplots(2, 2)
    plt.suptitle("Functions 2a-d over total x domain")
    for i, func in enumerate(func_list):
        axs[i % 2, i // 2].plot(total_x, func(total_x))
        axs[i % 2, i // 2].set_title(f"{func=}")

    print()
    # Assignment 2a
    x = np.linspace(2.5, 4.0, num=1000)
    for algorithm in algorithm_list:
        root, number_of_iterations = algorithm(func2a, x[0], x[-1], abs_err=1e-6, rel_err=1e-6)
        print(f"{algorithm=}, {root=}, {number_of_iterations=}")

    print()
    # Assignment 2b
    x = np.linspace(0, 0.48, num=1000)
    for algorithm in algorithm_list:
        root, number_of_iterations = algorithm(func2b, x[0], x[-1], abs_err=1e-6, rel_err=1e-6)
        print(f"{algorithm=}, {root=}, {number_of_iterations=}")

    print()
    # Assignment 2c
    x = np.linspace(-2.0, 0.0, num=1000)
    for algorithm in algorithm_list:
        root, number_of_iterations = algorithm(func2c, x[0], x[-1], abs_err=1e-6, rel_err=1e-6)
        print(f"{algorithm=}, {root=}, {number_of_iterations=}")

    print()
    # Assignment 2d
    x = np.linspace(0.0, 1.5, num=1000)
    for algorithm in algorithm_list:
        root, number_of_iterations = algorithm(func2d, x[0], x[-1], abs_err=1e-6, rel_err=1e-6)
        print(f"{algorithm=}, {root=}, {number_of_iterations=}")

    plt.show()


main()
