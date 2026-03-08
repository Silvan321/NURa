# Root finding

from collections.abc import Callable
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
    if fa < fb:
        fl, fh = fa, fb  # Set low and high function evaluation to fa and fb respectively if fa is smaller than fb, vice versa otherwise.
    else:  # This helps with readibility, but we don't want to waste function evaluations on this
        fl, fh = fb, fa

    fm = -1
    iteration = 0
    while np.isclose(fm, 0.0, rtol=rel_err, atol=abs_err) or iteration < max_number_of_iterations:
        c = (a + b) / 2
        fm = func(c)  # middle function evaluation
        if fm * fl < 0:  # root is in between low and middle value
            fh = fm  # update function evaluation
            b = c  # update x value to evaluate function
        else:
            fl = fm
        iteration += 1
