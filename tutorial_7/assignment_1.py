# Minimization and Maximization


import math
from collections.abc import Callable

import numpy as np
from matplotlib import pyplot as plt


def func1a(x):
    return x**4 + 10 * x**3 + 10 * (x - 2) ** 2


def bracket_minimum(func: Callable, a: float, b: float) -> tuple[float, float, float]:
    """Use the algorithm on lecture 7 slide to find a a 3 point bracket for a function bracketing exactly 1 minimum"""
    phi = (1 + math.sqrt(5)) / 2
    fa, fb = func(a), func(b)
    if fb < fa:  # Step 1: ensure f(b) < f(a)
        fa, fb = fb, fa
        a, b = b, a
    c = b + (b - a) * phi  # Step 2
    if (fc := func(c)) > fb:  # Step 3a
        return (a, b, c)
    # Step 3b: fit a polynomial. for this we can use LU Decomposition of a Vandermonde matrix
    # OR (since the minimum of the polynomial can be analytically defined using a,b,c and fa,fb,fc), we use the formula for this on slide 11 (Brent's method)
    d = b - 0.5 * (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa) / ((b - a) * (fb - fc) - (b - c) * (fb - fa))
    return


def main():
    x = np.linspace(-10, 10, num=100)

    plt.figure()
    plt.plot(x, func1a(x))
    plt.show()

    bracket_minimum(func=func1a, a=-9, b=-7)


main()
