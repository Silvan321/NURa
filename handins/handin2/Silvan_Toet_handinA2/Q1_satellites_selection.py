#### Sorting and Selection block ####


from matplotlib import pyplot as plt
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
    Note that the set values for a and c have been set manually to satisfy Hull-Dobell for an array of size 10000.

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
    return lcg(a=21, c=37, m=arr.size, size=size)
