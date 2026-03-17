#### Sorting and Selection block ####


from collections.abc import MutableSequence
from copy import deepcopy

import numpy as np
from Q1_satellites_sampling import lcg


def swap(a, i, j):
    """Helper function to swap elements a[i] and a[j]"""
    a[i], a[j] = a[j], a[i]


def selection_sort(A: MutableSequence, return_index: bool = False):
    """Implement the selection sorting algorithm.
    Based on Lecture 6 slide 6.
    If return_index is True, the sorted index array is returned.
    """
    a = deepcopy(A)
    N = len(a)
    if return_index:
        index_a = np.arange(N)
    for i in range(N - 1):
        i_min = i
        for j in range(i + 1, N):
            if a[j] < a[i_min]:
                i_min = j
        if i_min != i:
            swap(a, i, i_min)
            if return_index:
                swap(index_a, i, i_min)
    if return_index:
        return a, index_a
    return a


def quicksort(A: MutableSequence, return_index: bool = False):
    """Implement the quicksort sorting algorithm.
    Based on Lecture 6 slide 9.
    """
    a: MutableSequence = deepcopy(A)
    N = len(a)
    start_indx_fml = [0, N // 2, N - 1]
    if return_index:
        index_a = np.arange(N)
    first_middle_last, indx_fml = selection_sort(
        [a[start_indx_fml[0]], a[start_indx_fml[1]], a[start_indx_fml[2]]], return_index=True
    )  # take the median of the first, last and middle element as the pivot
    a[start_indx_fml[0]], a[start_indx_fml[1]], a[start_indx_fml[2]] = first_middle_last[0], first_middle_last[1], first_middle_last[2]
    if return_index:
        index_a[start_indx_fml[0]], index_a[start_indx_fml[1]], index_a[start_indx_fml[2]] = indx_fml[0], indx_fml[1], indx_fml[2]

    def recursive_part(sub_a: MutableSequence, sub_start_index: int):
        """This inner function is called recursively to sort smaller and smaller subarrays.
        sub_start_index tells the inner function where this specific subarray started, and therefore where its sorted values should be placed in the total array.
        """
        N = len(sub_a)
        x_pivot = sub_a[N // 2]  # since N will become smaller once we apply this step recursively, we don't hardcode this
        i = 0
        j = N - 1

        conda, condb = sub_a[i] >= x_pivot, sub_a[j] <= x_pivot  # Define these conditions as variables to avoid evaluating them unnecessarily many times!
        if conda and condb:  # Conditions might already be True in initial state, swap if so
            swap(sub_a, i, j)
            conda, condb = False, False
        while (not conda) or (not condb):
            # i goes up from 0 until a[i]>=x_pivot
            if not conda:
                i += 1
                conda = sub_a[i] >= x_pivot
            # j goes down from N-1 until a[j]<=x_pivot
            if not condb:
                j -= 1
                condb = sub_a[j] <= x_pivot
            # pointers have crossed. If we don't check this we will undo all our hard work: the elements that we have swapped will be swapped back. Also, size-2 subarrays will flip wrongly!
            if j <= i:
                break
            # once we meet both conditions, swap the elements and set conditions to false to continue this iteration of the loop
            if conda and condb:
                swap(sub_a, i, j)
                conda, condb = False, False

        sub_a_low, sub_a_high = sub_a[0:i], sub_a[i:N]
        if len(sub_a_low) == 1 and len(sub_a_high) == 1:
            # put at right position in final array
            a[sub_start_index : sub_start_index + 1] = sub_a_low
            a[sub_start_index + 1 : sub_start_index + 2] = sub_a_high
        elif len(sub_a_low) == 1:
            a[sub_start_index : sub_start_index + 1] = sub_a_low
            recursive_part(sub_a_high, sub_start_index + 1)
        elif len(sub_a_high) == 1:
            a[sub_start_index + i : sub_start_index + i + 1] = sub_a_high
            recursive_part(sub_a_low, sub_start_index)
        else:
            recursive_part(sub_a_low, sub_start_index), recursive_part(sub_a_high, sub_start_index + i)
        return sub_a

    a = recursive_part(a, 0)
    return a


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
