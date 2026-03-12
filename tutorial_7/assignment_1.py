# Minimization and Maximization


import math
from collections.abc import Callable

import numpy as np
from matplotlib import pyplot as plt


def func1a(x):
    return x**4 + 10 * x**3 + 10 * (x - 2) ** 2


def bracket_minimum(func: Callable, a: float, b: float) -> tuple[float, float, float]:
    """Use the algorithm on lecture 7 slide to find a a 3 point bracket for a function.
    Note that depending on the function and initial guess for a and b, this function is not guaranteed to find a bracket for only 1 minimum!
    There is no way to know this without evaluating the function at many points. This is why we have to restart often.
    """
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    fa, fb = func(a), func(b)
    if fb > fa:  # Step 1: ensure f(b) < f(a)
        fa, fb = fb, fa
        a, b = b, a
    c = b + (b - a) * phi  # Step 2
    while True:
        if (fc := func(c)) > fb:  # Step 3a
            return (a, b, c)
        # Step 3b: fit a polynomial. for this we can use LU Decomposition of a Vandermonde matrix
        # OR (since the minimum of the polynomial can be analytically defined using a,b,c and fa,fb,fc), we use the formula for this on slide 11 (Brent's method)
        d = b - 0.5 * (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa) / ((b - a) * (fb - fc) - (b - c) * (fb - fa))
        fd = func(d)
        if b < d < c:
            if fd < fc:  # Step 4
                return (b, d, c)
            if fd > fb:
                return (a, b, d)
            d = c + (c - b) * phi
        else:  # Step 5
            d = c + (c - b) * phi
        a, b, c = b, c, d
        fa, fb = fb, fc


def golden_section_search(func: Callable, a: float, b: float, c: float, target_acc: float = (np.finfo(float).eps) ** 0.5) -> float:
    """Iteratively tighten a 3 point bracket surrounding a minimum. This function assumes there is only 1 minimum inside the bracket.
    Use the algorithm defined in lecture 7 slide 10.
    """
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    w = 2 - phi
    c_min_b, b_min_a = abs(c - b), abs(b - a)
    other_edge = c if c_min_b > b_min_a else a  # Step 1
    while True:
        d = b + (other_edge - b) * w
        fb, fd = func(b), func(d)
        if abs(c - a) < target_acc:  # Step 2
            if fd < fb:
                return d
            return b
        if fd < fb:  # Step 3
            if b < d < c:
                a, b = b, d
                c_min_b, b_min_a = abs(c - b), abs(b - a)  # I need more lines for the various cases,
                other_edge = c if c_min_b > b_min_a else a  # but it is more efficient because I do not always have to recompute both intervals and check which value should become other_edge
            else:
                c, b = b, d
                c_min_b, b_min_a = abs(c - b), abs(b - a)
                other_edge = c if c_min_b > b_min_a else a
        else:  # Step 4
            if b < d < c:
                c = d
                c_min_b = abs(c - b)  # Step 5 optimization: only recompute tightened interval. Here larger interval is b_min_a
                other_edge = a
            else:
                a = d
                b_min_a = abs(b - a)
                other_edge = c


def main():
    x = np.linspace(-10, 10, num=100)

    plt.figure()
    plt.plot(x, func1a(x))
    plt.show()

    bracket_1a = bracket_minimum(func=func1a, a=-9, b=-3)
    print(f"{bracket_1a=}")

    golden_minimum_1a = golden_section_search(func1a, bracket_1a[0], bracket_1a[1], bracket_1a[2])
    print(f"{golden_minimum_1a=}")


main()
