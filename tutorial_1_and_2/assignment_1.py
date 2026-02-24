import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial


# Assignment 1a
# We are dealing with truncation error when the accuracy of the answer depends on the number of terms included in the power series approximation: where do we 'truncate', i.e. stop the power series?
def sinc_power_series(x, N: int):
    """Approximate the sinc function with its power series up to order N"""
    if isinstance(x, (int, np.float64, np.float32)):
        return np.sum(
            np.fromiter(
                (((-1) ** n * x ** (2 * n)) / factorial(2 * n + 1) for n in range(N)),
                dtype=np.float64,
            )
        )  # if x is a single value, we can directly return the approximation
    if isinstance(x, np.ndarray):
        y = np.empty_like(x, dtype=np.float64)
        for index, value in enumerate(x):
            y[index] = np.sum(
                np.fromiter(
                    (((-1) ** n * value ** (2 * n)) / factorial(2 * n + 1) for n in range(N)),
                    dtype=np.float64,
                )
            )
        return y
    raise TypeError("x must be a scalar or a numpy array")


def sin_x_over_x(x):
    if isinstance(x, (int, np.float64, np.float32)):
        return np.sin(x) / x if x != 0 else 1.0
    if isinstance(x, np.ndarray):
        y = np.empty_like(x, dtype=np.float64)
        for index, value in enumerate(x):
            y[index] = np.sin(value) / value if value != 0 else 1.0
        return y
    raise TypeError("x must be a scalar or a numpy array")


scalar_x = np.float64(2)
N = 10
sinc_approx = sinc_power_series(scalar_x, N)
sin_x_o_x = sin_x_over_x(scalar_x)
# Why not use np.sinc? Because np.sinc is normalized, meaning that np.sinc(x) = sin(pi*x)/(pi*x), so we would have to divide by pi to get the unnormalized sinc value,
# which is what we are approximating with the power series.
sinc_numpy = np.sinc(scalar_x / np.pi)
print(f"Approximation: {sinc_approx}, Numpy sin_x_over_x: {sin_x_o_x}, Numpy sinc: {sinc_numpy}")
print(f"Error of approximation when type of x is {scalar_x.dtype} and N is {N}: {np.abs(sinc_approx - sin_x_o_x)}")


def plot_sincs(x: NDArray[np.double], N: int):
    plt.figure()
    plt.title("Approximation and exact")
    plt.plot(x, sinc_power_series(x, N), label=f"Sinc Power Series Approximation (N={N})")
    plt.plot(x, sin_x_over_x(x), label="Sinc Function")
    plt.legend()
    plt.figure()
    plt.title("Error between approximation and exact")
    plt.plot(x, np.abs(sinc_power_series(x, N) - sin_x_over_x(x)))
    plt.show()


x = np.arange(-10, 10, 0.01, dtype=np.float64)
# plot_sincs(x, N=11)
# Assignment 1b
# The plot shows that the power series approximation is very close to the actual sinc function for small values of x, but as x gets larger, the approximation diverges from the actual function.
# This is because the power series is only accurate near x=0, and as we move further away from this point, the error increases.
# As we increase N, the number of terms, the error becomes smaller. The error also oscillates: for positive x, the approximation is above the actual function,
# and for negative x, the approximation is below the actual function. This is because the power series is an alternating series, meaning that the terms alternate in sign,
# which causes the approximation to oscillate around the actual function.

# Assignment 1c
# The power series approximation achieves good accuracy with only a few terms when x is close to 0.
# When we switch from np.float64 to np.float32 with N=20 at x=2, the error jumps up from 1.1102230246251565e-16 to 1.0076125267488578e-08
# This discrepancy is due to a larger roundoff error in x for np.float32 which is enhanced (to the power ...) and summed in the power series.
# This is the minimal error that we can get with np.float32 (single precision) numbers, and we reach it already after 8 terms in the power series expansion approximation
# Remember: the roundoff error, caused by the limited machine precision, is only a problem when we have to combine numbers with widely varying magnitudes: at this point we have to shift the bits
# from the smaller number into oblivion.
scalar_x = np.float32(2)
for n in range(20):
    sinc_approx = sinc_power_series(scalar_x, n)
    sin_x_o_x = sin_x_over_x(scalar_x)
    # Why not use np.sinc? Because np.sinc is normalized, meaning that np.sinc(x) = sin(pi*x)/(pi*x), so we would have to divide by pi to get the unnormalized sinc value,
    # which is what we are approximating with the power series.
    sinc_numpy = np.sinc(scalar_x / np.pi)
    print(f"Approximation: {sinc_approx}, Numpy sin_x_over_x: {sin_x_o_x}, Numpy sinc: {sinc_numpy}")
    print(f"Error of approximation when type of x is {scalar_x.dtype} and N is {n}: {np.abs(sinc_approx - sin_x_o_x)}")
