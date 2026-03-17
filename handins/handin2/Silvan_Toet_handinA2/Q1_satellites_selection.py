#### Sorting and Selection block ####


import numpy as np
from Q1_satellites_sampling import lcg


def sort_array(
    arr: np.ndarray,
    inplace: bool = False,
) -> np.ndarray:
    """Sort a 1D array using a sorting algorithm of your choice

    Parameters
    ----------
    arr : ndarray
        Input array to be sorted
    inplace : bool, optional
        If True, sort the array in-place
        If False, return a sorted copy

    Returns
    -------
    sorted_arr : ndarray
        Sorted array (same shape as arr)

    """
    if inplace:
        sorted_arr = arr
    else:
        sorted_arr = arr.copy()

    # TODO: sort sorted_arr in-place here

    return sorted_arr


def choice(arr: np.ndarray, size: int = 1) -> np.ndarray:
    """Choose given number of random elements from an array, without replacement
    We use the Hull-Dobell theorem for Linear Congruential Generators, which ensures all values up to the period m are generated exactly once before the sequence repeats.
    Then we scale the generated values to the size of the array, and use the generated values (converted to ints) as indices from the array
    Parameters
    ----------
    arr : ndarray
        Array to shuffle
    size : int, optional
        Number of elements to pick from array
        The default is 1

    Returns
    -------
    chosen : ndarray
        Randomly chosen elements from arr, shape (size,)
    """
    # first generate 100 random values without duplicates
    m = 1 << size.bit_length()  # set m to the next power of two larger than size
    P_uniform = lcg(a=5, c=1, m=m, size=size)  # Since m is a power of two, the values of a and c can stay 5 and 1 respectively and still satisfy Hull-Dobell
    P_indices = int(
        arr.size * P_uniform
    )  # Scale the generated random numbers to the size of the array to sample from, in this case from the array with 10000 generated galaxies in question b. Then convert to int for indices
    return arr[P_indices]
