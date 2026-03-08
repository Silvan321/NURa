# Sorting and Selection

from collections.abc import MutableSequence
from copy import deepcopy
from math import e
from timeit import timeit

from matplotlib import pyplot as plt
import numpy as np


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

        sub_a_low, sub_a_high = sub_a[0:i], sub_a[i:N]  # FIX THIS SELECTION CALL FOR LENGTH 2 ARRAYS!
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


def main():
    # Assignment 1a
    np.random.seed(1)
    A = np.random.randint(0, 100, size=100)  # [3, 27, 43, 10, 9, 82, 38]  # np.random.randint(0, 100, size=20)
    # Assignment 1a: Selection sort
    # The disadvantage of this algorithm is it uses O(N^2) comparisons (and O(N) swaps)
    # It is also not stable (although I would think that the first occurence of certain value is placed at the front)
    A_selection_sorted = selection_sort(A, return_index=True)
    print(A)
    print(A_selection_sorted)

    # Assignment 1b
    A_quicksorted = quicksort(A)
    print(A_quicksorted)

    # Assignment 1c
    N_bounds = (5, 1e4)
    N_bounds_log10 = np.log(N_bounds)
    number_of_values = 6
    N_values_log10 = np.linspace(start=N_bounds_log10[0], stop=N_bounds_log10[1], num=number_of_values)
    N_values = [int(N) for N in e**N_values_log10]
    x = range(number_of_values)

    print(f"{N_values=}")
    plt.figure()
    plt.title("N values equally spaced in log space\n We only go up to 1e4 to prevent the runtime for selection sort taking forever")
    plt.scatter(x, N_values)
    plt.yscale("log")
    plt.show()

    # compute times
    computation_number = 1

    sorting_algorithms = [selection_sort, quicksort]
    timing_array = np.zeros((len(N_values), len(sorting_algorithms)))

    for i, N in enumerate(N_values):
        A = np.random.randint(0, 100, size=N)
        for j, sorting_algorithm in enumerate(sorting_algorithms):
            time = timeit(stmt=lambda: sorting_algorithm(A), number=computation_number) / computation_number
            timing_array[i, j] = time
            print(f"{N=}, {sorting_algorithm=} combinations takes {time} seconds")
        print()

    # We see that selection sort is faster when N is relatively small (N<~50).
    # When N becomes larger than 100, quicksort quickly becomes orders of magnitude faster!

    # Assignment 1d

    A_size = len(A_quicksorted)
    percentiles = [16, 50, 84]
    for percentile in percentiles:
        A_percentile = A_quicksorted[int(percentile * A_size / 100)]
        print(f"{percentile=}, {A_percentile=}")


main()
