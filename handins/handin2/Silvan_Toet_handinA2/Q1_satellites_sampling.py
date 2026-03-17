#### Sampler block including RNGs ####

from collections.abc import Callable, Iterable

import numpy as np


def rng_64bit_xor_shift(x: np.uint64 = np.uint64(123456789), a_iter: Iterable[np.uint64] = (np.uint64(21), np.uint64(35), np.uint64(4)), size: int = 1, scale_uniform: bool = False):
    """Use the 64-bit XOR-shift method for making a Random Number Generator on slide 14 of lecture 5.
    x is the seed, which should be an unsigned 64 bit integer, not 0.
    We use a numpy variable, as Python built-in variables will automatically extend if bits are moved out of range (which is what we want to use here: moving bits out of range hides info for the user).
    a is an iterable of ints containing the coefficients that will be used to bitshift x by a certain amount before doing an XOR operation with itself.
    Size is the number of random numbers generated. scale_uniform scales the output to the range [0,1]
    If scale_uniform is False, the output is in the range (0, 2^32-1), since we only supply the lowest 32 bits of each generated random number to the user to hide info about the state of the RNG.
    """
    x = np.uint64(x)
    a_list = [np.uint64(a) for a in a_iter]  # Convert all coefficients to np.uint64 variables once, to ensure no error occurs if the user does not supply these in np.uint64 format
    if x == 0:
        raise ValueError("x cannot have 0 as a starting value")

    random_values_array = np.zeros(size, dtype=np.uint32)
    for i in range(len(random_values_array)):
        for j, a in enumerate(a_list):
            x = x ^ (x >> a) if j % 2 == 0 else x ^ (x << a)  # Even coefficients are used to shift x a bits to the right, odd coefficients are used to shift x a bits to the left
            # This is necessary to ensure randomness: otherwise we would only be changing e.g. the lower significant bits, and numbers would oscillate around the starting seed value
        random_values_array[i] = np.uint32(x)  # We use a bitmask to provide the user with only the lowest 32 bits: restrict information about the full state to the user!
    if scale_uniform:
        return random_values_array / np.iinfo(np.uint32).max  # Divide by max np.int32 value because we only supply the lowest 32 bits to the user.
    # Division operator converts dtypes to np.float64, which is what we want here
    return random_values_array


def lcg(x: np.uint64 = np.uint64(123456789), a: np.uint64 = np.uint64(7654321), c: np.uint64 = np.uint64(3333333), m: np.uint64 = np.uint64(1234543212345), size: int = 1, scale_uniform: bool = False):
    """Implement a (Multiple) Linear Congruential Generator.
    When c=0 we are dealing with a MLCG.
    This RNG is bad on its own but a good building block for more complex RNGs.
    Size is the number of random numbers generated. scale_uniform scales the output to the range [0,1]
    If scale_uniform is False, the output is in the range (0, 2^32-1), since we only supply the lowest 32 bits of each generated random number to the user to hide info about the state of the RNG.
    """
    x, a, c, m = np.uint64(x), np.uint64(a), np.uint64(c), np.uint64(m)  # Cast to unsigned integers should user forget to do so
    random_values_array = np.zeros(size, dtype=np.uint32)
    for i in range(len(random_values_array)):
        x = (a * x + c) % m
        random_values_array[i] = np.uint32(x)
    if scale_uniform:
        return random_values_array / np.iinfo(np.uint32).max
    return random_values_array


def additive_combined_rng(
    x1: np.uint64 = np.uint64(123456789),
    x2: np.uint64 = np.uint64(987654321),
    a_iter: Iterable[np.uint64] = (np.uint64(21), np.uint64(35), np.uint64(4)),
    a: np.uint64 = np.uint64(7654321),
    c: np.uint64 = np.uint64(3333333),
    m: np.uint64 = np.uint64(1234321),
    size: int = 1,
    scale_uniform: bool = True,
):
    """Combine two RNGs together. In this case a 64 bit XOR shift RNG and a Linear Congruential Generator.
    Default values have been set to values that seem to generate random results.
    """
    P_64 = rng_64bit_xor_shift(x=x1, a_iter=a_iter, size=size, scale_uniform=scale_uniform)
    P_lcg = lcg(x=x2, a=a, c=c, m=m, size=size, scale_uniform=scale_uniform)
    P_add = P_64 + P_lcg
    if scale_uniform:
        return P_add / 2  # Both sub generator outputs have been scaled to [0,1) range, so here we need to divide by 2
    return P_add


def sampler(dist: Callable, a: float, b: float, Nsamples: int) -> np.ndarray:
    """Sample from a normalized distribution using rejection sampling.

    Parameters
    ----------
    dist : callable
        Distribution to sample
    a :
        Minimum value for sampling
    b : float
        Maximum value for sampling
    Nsamples : int
        Number of samples

    Returns
    -------
    sample: ndarray
        Values sampled from dist, shape (Nsamples,)
    """
    samples = np.zeros(Nsamples)
    number_of_acquired_samples = 0
    while number_of_acquired_samples < Nsamples:  # Since we might reject a lot of samples, keep generating samples in batches of Nsamples until we have enough
        P1_uniform = additive_combined_rng(size=Nsamples)  # Generate random numbers for the abscissa range of the distribution that we to sample from
        P_ab = a + (b - a) * P1_uniform  # Scale these to the abscissa range
        P2_uniform = additive_combined_rng(size=Nsamples)  # Then generate a separate random number from a uniform distribution [0,1), which we interpret as the probability
        for x, y in zip(  # noqa: B905
            P_ab, P2_uniform
        ):  # Then we accept x into our sample if y < p(x): the higher p(x) is for that value of x, the more likely x should be, and therefore the more likely that y is smaller than p(x)
            if y < dist(x):
                samples[number_of_acquired_samples] = x
                number_of_acquired_samples += 1
            if number_of_acquired_samples >= Nsamples:
                break

    return samples
