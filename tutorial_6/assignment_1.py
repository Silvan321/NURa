# Sorting and Selection

import numpy as np
from copy import deepcopy

import random


def selection_sort(A: np.ndarray):
    """Implement the selection sorting algorithm.
    a should be a 1D array"""
    if A.ndim > 1:
        raise ValueError("a should be a 1 dimensional array")
    a = deepcopy(A)
    N = len(a)
    for i in range(N - 1):
        i_min = i
        for j in range(i + 1, N):
            if a[j] < a[i_min]:
                i_min = j
        if i_min != i:
            a[i], a[i_min] = a[i_min], a[i]
    return a


def main():
    A = np.random.randint(0, 100, size=100)
    A_selection_sorted = selection_sort(A)
    print(A_selection_sorted)


main()
