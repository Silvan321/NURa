# Sorting and Selection

from collections.abc import Iterable, Sequence
from copy import deepcopy

import numpy as np


def swap(a, i, j):
    a[i], a[j] = a[j], a[i]


def selection_sort(A: Sequence, return_index: bool = False):
    """Implement the selection sorting algorithm.
    If return_index is True, the sorted index array is returned
    """
    a = deepcopy(A)
    N = len(a)
    index_a = np.arange(N)
    for i in range(N - 1):
        i_min = i
        for j in range(i + 1, N):
            if a[j] < a[i_min]:
                i_min = j
        if i_min != i:
            swap(a, i, i_min)
            swap(index_a, i, i_min)
    if return_index:
        return a, index_a
    return a


def quicksort(A: Sequence):
    """Implement the quicksort sorting algorithm."""
    a = deepcopy(A)
    N = len(a)
    indx_fml = [0, N // 2, N - 1]
    first_middle_last, start_indices = selection_sort([a[indx_fml[0]], a[indx_fml[1]], a[indx_fml[2]]], return_index=True)  # take the median of the first, last and middle element as the pivot
    a[indx_fml[0]], a[indx_fml[1]], a[indx_fml[2]] = first_middle_last[0], first_middle_last[1], first_middle_last[2]
    for i, start_index in enumerate(start_indices):
        if start_index == 1:
            start_indices[i] = N // 2
        if start_index == 2:
            start_indices[i] = N - 1

    for first_middle_last_index, start_index in zip(indx_fml, start_indices):
        if first_middle_last_index != start_index:
            swap(a, first_middle_last_index, start_index)

    return first_middle_last, start_indices


def main():
    np.random.seed(1)
    A = np.random.randint(0, 100, size=100)
    # Assignment 1a: Selection sort
    # The disadvantage of this algorithm is it uses O(N^2) comparisons (and O(N) swaps)
    # It is also not stable (although I would think that the first occurence of certain value is placed at the front)
    A_selection_sorted = selection_sort(A, return_index=True)
    print(A)
    print(A_selection_sorted)

    A_quicksorted = quicksort(A)
    print(A_quicksorted)


main()
