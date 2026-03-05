# Sorting and Selection

from collections.abc import Iterable, Sequence
from copy import deepcopy

import numpy as np


def swap(a, i, j):
    """Helper function to swap elements a[i] and a[j]"""
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

    def recursive_part(sub_a):
        """This inner function is called recursively to sort smaller and smaller subarrays."""
        N = len(sub_a)
        x_pivot = sub_a[N // 2]  # since N will become smaller once we apply this step recursively, we don't hardcode this
        i = 0
        j = N - 1
        i_stop, j_stop = None, None
        while (small := sub_a[i]) < x_pivot or (big := sub_a[j]) > x_pivot:
            # We want to run one loop until a[i] >= x_pivot AND a[j] <= x_pivot. Use the walrus operator to prevent evaluating a[i] and a[j] two (OR MORE!) times for the same value of i/j
            # That means the below if and elif statement cannot both be True if the while loop is still running
            # second part of or statement not evaluated when first part is true!
            if small < x_pivot:
                i += 1
            else:
                i_stop = i
            if big > x_pivot:
                j -= 1
            else:
                j_stop = j
            if (i_stop is not None) and (j_stop is not None):
                swap(sub_a, i_stop, j_stop)

        conda, condb = False, False
        while (not conda) or (not condb):
            if not conda:
                i += 1
                conda = sub_a[i] >= x_pivot
            if not condb:
                j += 1
                condb = sub_a[j] <= x_pivot
            if conda and condb:
                swap(sub_a, i, j)

        sub_a_low, sub_a_high = sub_a[0 : N // 2], sub_a[N // 2 + 1, N]
        # This will fail if the subarray is 2 in size
        if len(sub_a_low) == 1 and len(sub_a_high) == 1:
            # put at right position in final array
            pass
        elif len(sub_a_low) == 1:
            # put at right position in final array
            recursive_part(sub_a_high)
        elif len(sub_a_high) == 1:
            # put at right position in final array
            recursive_part(sub_a_low)
        else:
            recursive_part(sub_a_low), recursive_part(sub_a_high)
        return sub_a

    a = recursive_part(a)
    return a


# keep indices of full array, to pass to recursive arrays


def main():
    np.random.seed(1)
    A = np.random.randint(0, 100, size=10)
    # Assignment 1a: Selection sort
    # The disadvantage of this algorithm is it uses O(N^2) comparisons (and O(N) swaps)
    # It is also not stable (although I would think that the first occurence of certain value is placed at the front)
    A_selection_sorted = selection_sort(A, return_index=True)
    print(A)
    print(A_selection_sorted)

    A_quicksorted = quicksort(A)
    print(A_quicksorted)


main()
