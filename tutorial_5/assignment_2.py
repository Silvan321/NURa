# Random Number Generators
# RNG "States" are based on all info that the RNG has available.
# The RNG should not reveal its entire info to the user, otherwise the user can predict the next number
# A and B are sub RNGs
# A output not based on B output, but uses intermediate output
# A and B both use their own seed


from math import pi
from collections.abc import Iterable

import numpy as np
from matplotlib import pyplot as plt


def rng_64bit_xor_shift(x: np.uint64 = np.uint64(123456789), a_iter: Iterable[np.uint64] = (np.uint64(21), np.uint64(35), np.uint64(4)), size: int = 1, scale_uniform: bool = False):
    """Use the 64-bit XOR-shift method for making a Random Number Generator on slide 14 of lecture 5.
    x is the seed, which should be an unsigned 64 bit integer, not 0.
    We use a numpy variable, as Python built-in variables will automatically extend if bits are moved out of range (which is what we want to use here: moving bits out of range hides info for the user).
    a is an iterable of ints containing the coefficients that will be used to bitshift x by a certain amount before doing an XOR operation with itself.
    Size is the number of random numbers generated. scale_uniform scales the output to the range [0,1]
    """
    x = np.uint64(x)
    a_list = [np.uint64(a) for a in a_iter]  # Convert all coefficients to np.uint64 variables once, to ensure no error occurs if the user does not supply these in np.uint64 format
    if x == 0:
        raise ValueError("x cannot have 0 as a starting value")

    random_values_array = np.zeros(size, dtype=np.uint64)
    for i in range(len(random_values_array)):
        for j, a in enumerate(a_list):
            x = x ^ (x >> a) if j % 2 == 0 else x ^ (x << a)  # Even coefficients are used to shift x a bits to the right, odd coefficients are used to shift x a bits to the left
            # This is necessary to ensure randomness: otherwise we would only be changing e.g. the lower significant bits, and numbers would oscillate around the starting seed value
        random_values_array[i] = x
    if scale_uniform:
        return random_values_array / np.iinfo(np.int64).max  # Division operator converts dtypes to np.float64, which is what we want here
    return random_values_array


def lcg(x: np.uint64 = np.uint64(123456789), a: np.uint64 = np.uint64(7654321), c: np.uint64 = np.uint64(3333333), m: np.uint64 = np.uint64(1234543212345), size: int = 1, scale_uniform: bool = False):
    """Implement a (Multiple) Linear Congruential Generator.
    When c=0 we are dealing with a MLCG.
    This RNG is bad on its own but a good building block for more complex RNGs.
    Size is the number of random numbers generated. scale_uniform scales the output to the range [0,1]
    """
    x, a, c, m = np.uint64(x), np.uint64(a), np.uint64(c), np.uint64(m)  # Cast to unsigned integers should user forget to do so
    random_values_array = np.zeros(size, dtype=np.uint64)
    for i in range(len(random_values_array)):
        x = (a * x + c) % m
        random_values_array[i] = x
    if scale_uniform:
        return random_values_array / np.iinfo(np.int64).max
    return random_values_array


def additive_combined_rng(
    x1: np.uint64 = np.uint64(123456789),
    x2: np.uint64 = np.uint64(987654321),
    a_iter: Iterable[np.uint64] = (np.uint64(21), np.uint64(35), np.uint64(4)),
    a: np.uint64 = np.uint64(7654321),
    c: np.uint64 = np.uint64(3333333),
    m: np.uint64 = np.uint64(1234543212345),
    size: int = 1,
    scale_uniform: bool = True,
):
    P_64 = rng_64bit_xor_shift(x1, size=size, scale_uniform=scale_uniform)
    P_lcg = lcg(x2, size=size, scale_uniform=scale_uniform)
    P_add = P_64 + P_lcg
    if scale_uniform:
        return P_add / 2  # Both sub generator outputs have been scaled to [0,1) range, so here we need to divide by 2
    return P_add


def theta_phi_1a(P1, P2):
    theta = pi * P1
    phi = 2 * pi * P2
    return theta, phi


def theta_phi_1b(P1, P2):
    theta = np.arccos(1 - 2 * P1)
    phi = 2 * pi * P2
    return theta, phi


def main():
    num_points = 1000
    rng_size = 10000
    # We need TWO DIFFERENT random number generators, otherwise every theta, phi pair uses the same random number
    # Then we get a helix with the shape theta = 2 phi
    P_64 = rng_64bit_xor_shift(size=rng_size, scale_uniform=True)
    print(P_64)

    P_lcg = lcg(size=rng_size, scale_uniform=True)
    print(P_lcg)

    P_add = additive_combined_rng(size=rng_size, scale_uniform=True)
    print(P_add)

    rngs_dict = {"64bit_XOR": P_64, "LCG": P_lcg, "Additive Combination": P_add}

    fig, axs = plt.subplots(3, 1)
    for i, key in enumerate(rngs_dict):
        axs[i].set_title(key)
        axs[i].hist(rngs_dict[key])

    x = np.arange(rng_size)
    fig, axs = plt.subplots(3, 1)
    for i, key in enumerate(rngs_dict):
        freqs = np.fft.fft(rngs_dict[key])
        axs[i].set_title(key)
        axs[i].scatter(x, freqs)

    P_u1 = np.random.uniform(0, 1, size=num_points)
    P_u2 = np.random.uniform(0, 1, size=num_points)
    r = np.ones(num_points)

    theta_1a, phi_1a = theta_phi_1a(P_u1, P_u2)
    x_1a = r * np.sin(theta_1a) * np.cos(phi_1a)
    y_1a = r * np.sin(theta_1a) * np.sin(phi_1a)
    z_1a = r * np.cos(theta_1a)

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(x_1a, y_1a, z_1a)

    theta_1b, phi_1b = theta_phi_1b(P_u1, P_u2)
    x_1b = r * np.sin(theta_1b) * np.cos(phi_1b)
    y_1b = r * np.sin(theta_1b) * np.sin(phi_1b)
    z_1b = r * np.cos(theta_1b)

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(x_1b, y_1b, z_1b)

    plt.show()
    # As we can see, the second option corrects for the fact that a surface element on a unit sphere is compressed near the poles
    # In the case of the theta phi generation in 1a, we get more values near the poles
    # Think of a globe: Meridian lines converge as you get closer to the poles


main()
