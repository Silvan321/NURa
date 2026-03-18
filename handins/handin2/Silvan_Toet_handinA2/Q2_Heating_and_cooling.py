from collections.abc import Callable
from functools import partial
from matplotlib import pyplot as plt
import numpy as np

# Constants (mind the units!)

psi = 0.929
Tc = 1e4  # K
Z = 0.015
k = 1.38e-16  # erg/K
aB = 2e-13  # cm^3 / s
A = 5e-10
xi = 1e-15


# There's no need for nH nor ne as they cancel out
def equilibrium1(T, Z, Tc, psi):
    return psi * Tc * k - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T * k  # Note: this is already Gamma_pe - Lambda_rr for the first question, with n_h and n_e cancelled out.


# Therefore we can equate this to 0 and simply find the zero-crossing aka root.


def equilibrium2(T, Z, Tc, psi, nH, A, xi, aB):
    return (psi * Tc - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T - 0.54 * (T / 1e4) ** 0.37 * T) * k * nH * aB + A * xi + 8.9e-26 * (T / 1e4)  # and here the same applies


# Derivative function, might be useful if using Newton-Raphson method for root finding
# def equilibrium2_deriv(T, nH):
#     # TODO: Compute derivative of equilibrium2 with respect to T
#     return 0.0


#### root finder ####


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
        c = (a + b) * 0.5
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
        # if ((a - c) * (c - b)) < 0:  # a should be smaller than c and c should be smaller than b while inside bracket, so both negative, so product positive
        #    raise ValueError("Error: Secant jumped out of bracket")
        fc = func(c)  # middle function evaluation
        iteration += 1
        b, a = c, b  # update b and a (x_i and x_i-1) to c and b (x_i+1 and x_i) respectively
        fa, fb = func(a), func(b)
    return c, iteration


def false_position(func: Callable, a: float, b: float, abs_err: float, rel_err: float, max_number_of_iterations: int = 50):
    """Implement the false position algorithm for finding a single root for a 1D function.
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
        if fa * fc < 0:  # root between a and c
            b = c
            fb = func(b)
        else:  # root between b and c
            b, a = c, b
            fa, fb = func(a), func(b)
        iteration += 1
    return c, iteration


def main():
    equilibrium1_partial = partial(equilibrium1, Z=Z, Tc=Tc, psi=psi)
    equilibrium2_partial = partial(equilibrium2, Z=Z, Tc=Tc, psi=psi, A=A, xi=xi, aB=aB)
    # Initial bracket
    bracket = (1, 1e7)

    x1_range = 10 ** np.linspace(np.log10(bracket[0]), np.log10(bracket[1]), num=1000)

    plt.figure(figsize=(16, 10))
    plt.title("Temperature distribution with only photoionization vs radiative recombination\n Equilibrium where y=0")
    plt.plot(x1_range, equilibrium1_partial(x1_range))
    plt.semilogx()
    plt.savefig("Plots/equilibrium1.png", dpi=600)

    plt.figure(figsize=(16, 10))
    plt.title("Temperature distribution with only photoionization vs radiative recombination\n Equilibrium where y=0 zoomed")
    plt.plot(x1_range, equilibrium1_partial(x1_range))
    plt.ylim(-1e-12, 1e-12)
    plt.semilogx()
    plt.savefig("Plots/equilibrium1_zoomed.png", dpi=600)
    plt.show()

    algorithm_list = [bisection, secant, false_position]

    with open("Calculations/equilibrium_temp_simple.txt", "w") as f:
        for algorithm in algorithm_list:
            root, number_of_iterations = algorithm(equilibrium1_partial, x1_range[0], x1_range[-1], abs_err=1e-14, rel_err=1e-12)
            f.write(f"root {root:.12g}, n_iter {number_of_iterations}\n")

    ### 2b ####

    # Initial bracket
    bracket = (1, 1e15)
    x2_range = 10 ** np.linspace(np.log10(bracket[0]), np.log10(bracket[1]), num=1000)

    ne_list = [1e-4, 1, 1e4]

    fig2a, axs = plt.subplots(1, 3, figsize=(16, 10))
    plt.suptitle("Temperature distributions for different values of n_e\n Equilibrium where y=0")
    for i, ne in enumerate(ne_list):
        axs[i].plot(x2_range, equilibrium2_partial(T=x2_range, nH=ne))
        axs[i].set_xscale("log")
    plt.savefig("Plots/equilibrium2.png", dpi=600)

    fig2a, axs = plt.subplots(1, 3, figsize=(16, 10))
    plt.suptitle("Temperature distributions for different values of n_e\n Equilibrium where y=0 zoomed")
    for i, ne in enumerate(ne_list):
        axs[i].plot(x2_range, equilibrium2_partial(T=x2_range, nH=ne))
        axs[i].set_xscale("log")
        axs[i].set_ylim(-1e-20, 1e-20)
    plt.savefig("Plots/equilibrium2_zoomed.png", dpi=600)

    # We again assume the gas is fully ionized, so for nH we use the given values for ne
    for nH in ne_list:
        print()
        for algorithm in algorithm_list:
            equilibrium2_with_nh = partial(equilibrium2_partial, nH=nH)
            root, number_of_iterations = algorithm(equilibrium2_with_nh, x2_range[0], x2_range[-1], abs_err=1e-14, rel_err=1e-12)
        if nH == 1e-4:
            with open("Calculations/equilibrium_low_density.txt", "w") as f:
                for algorithm in algorithm_list:
                    equilibrium2_with_nh = partial(equilibrium2_partial, nH=nH)
                    root, number_of_iterations = algorithm(equilibrium2_with_nh, x2_range[0], x2_range[-1], abs_err=1e-20, rel_err=1e-12)
                    f.write(f"root {root:.12g}, n_iter {number_of_iterations}\n")
        elif nH == 1:
            with open("Calculations/equilibrium_mid_density.txt", "w") as f:
                for algorithm in algorithm_list:
                    equilibrium2_with_nh = partial(equilibrium2_partial, nH=nH)
                    root, number_of_iterations = algorithm(equilibrium2_with_nh, x2_range[0], x2_range[-1], abs_err=1e-20, rel_err=1e-12)
                    f.write(f"root {root:.12g}, n_iter {number_of_iterations}\n")
        elif nH == 1e4:
            with open("Calculations/equilibrium_high_density.txt", "w") as f:
                for algorithm in algorithm_list:
                    equilibrium2_with_nh = partial(equilibrium2_partial, nH=nH)
                    root, number_of_iterations = algorithm(equilibrium2_with_nh, x2_range[0], x2_range[-1], abs_err=1e-20, rel_err=1e-12)
                    f.write(f"root {root:.12g}, n_iter {number_of_iterations}\n")


if __name__ == "__main__":
    main()
