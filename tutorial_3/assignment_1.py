from math import pi

import numpy as np


def trapezoid(a, b, func, N: int):
    """Use the extended trapezoid rule to calculate the integral.
    Parameters a and b are start and stop x values of the range to be integrated respectively.
    func is the function to be evaluated.
    N is the number of evaluations.
    """
    xdata = np.linspace(a, b, num=N)
    h = (b - a) / N  # step size
    return h * (0.5 * (func(b) + func(a)) + np.sum(func(xdata[1:-1])))


def simpson(a, b, func, N: int):
    """Use the extended Simpson's rule to calculate the integral.
    Parameters a and b are start and stop x values of the range to be integrated respectively.
    func is the function to be evaluated.
    N is the number of evaluations.
    """
    xdata = np.linspace(a, b, num=N)
    h = (b - a) / N  # step size
    _sum = 0
    for index, value in enumerate(xdata):
        if (index == 0) or (index == N - 1):
            _sum += func(value)
        elif index % 2 == 0:  # even terms
            _sum += 4 * func(value)
        else:  # odd terms except first and last
            _sum += 2 * func(value)
    return h / 3 * _sum


def simpson_vectorized(a, b, func, N: int):
    """Use the extended Simpson's rule to calculate the integral.
    Vectorized version with array slizing for optimization -> time the difference!
    Parameters a and b are start and stop x values of the range to be integrated respectively.
    func is the function to be evaluated.
    N is the number of evaluations.
    """
    xdata = np.linspace(a, b, num=N)
    h = (b - a) / N  # step size
    return h / 3 * (np.sum(4 * func(xdata[1::2])) + np.sum(2 * func(xdata[2:-1:2])) + func(xdata[0]) + func(xdata[-1]))


def romberg(a, b, func, N_start: int, order: int):
    """See slide 14 of Lecture 4 annotated slides for the formula used.
    order is the number of initial approximations.
    """
    romberg_table = np.zeros((order, order))
    for i in range(order):
        N = (2**i) * N_start  # number of intervals doubles with each depth
        romberg_table[i, 0] = trapezoid(a, b, func, N)
        for j in range(1, i + 1):
            romberg_table[i, j] = (4**j * romberg_table[i, j - 1] - romberg_table[i - 1, j - 1]) / (4**j - 1)  # General version of 4/3 S_1 - 1/3 S_0
    return romberg_table[order - 1, order - 1]


def romberg_vector_version(a, b, func, N_start: int, order: int):
    """See slide 14 of Lecture 4 annotated slides for the formula used.
    order is the number of initial approximations.
    """
    romberg_vector = np.zeros(shape=order)
    for j in range(order):
        for i in range(order - j):
            if j == 0:  # the first time we fill the column, we use the trapezoid rule with doubling step sizes per row entry
                N = (2**i) * N_start  # number of intervals doubles with each depth
                romberg_vector[i] = trapezoid(a, b, func, N)
            else:  # Subsequent times we use the update rule
                romberg_vector[i] = (4**j * romberg_vector[i + 1] - romberg_vector[i]) / (4**j - 1)  # General version of 4/3 S_1 - 1/3 S_0
    return romberg_vector[0]


def x_squared(x):
    return x * x


def q1d_func(x):
    return 3 * np.exp(-2 * x) + (x / 10) ** 4


def main():
    # Analytical i.e. exact answers
    x_squared_analytical = (5**3 - 1**3) / 3
    print(f"{x_squared_analytical=}")
    sin_x_analytical = 2
    print(f"{sin_x_analytical=}")

    # Numpy trapezoid
    N = 8
    print(f"{N=}")
    x_squared_x_data = np.linspace(1, 5, num=N)
    print(f"{np.trapz(x_squared_x_data**2, x_squared_x_data)=}")

    sin_x_x_data = np.linspace(0, pi, num=N)
    print(f"{np.trapz(np.sin(sin_x_x_data), sin_x_x_data)=}")
    print()

    # Assignment 1a

    x_squared_trapz = trapezoid(1, 5, x_squared, N)
    print(f"{x_squared_trapz=}")

    sin_x_trapz = trapezoid(0, pi, np.sin, N)
    print(f"{sin_x_trapz=}")

    # Assignment 1b
    x_squared_simp = simpson(1, 5, x_squared, 2 * N)
    print(f"{x_squared_simp=}")
    x_squared_simp_vec = simpson_vectorized(1, 5, x_squared, 2 * N)
    print(f"{x_squared_simp_vec=}")

    # Test vectorized vs non-vectorized version of Simpson's rule and time the difference
    # repeat_statement_number = 10
    # simpson_vectorized_time = timeit(lambda: simpson_vectorized(1, 5, x_squared, 2 * N), number=repeat_statement_number)
    # print(f"{simpson_vectorized_time=} seconds")
    # simpson_normal_time = timeit(lambda: simpson(1, 5, x_squared, 2 * N), number=repeat_statement_number)
    # print(f"{simpson_normal_time=} seconds")

    sin_x_simp = simpson(0, pi, np.sin, 2 * N)
    print(f"{sin_x_simp=}")
    sin_x_simp_vec = simpson_vectorized(0, pi, np.sin, 2 * N)
    print(f"{sin_x_simp_vec=}")

    # compare Simpsons method with 4/3 S_1 - 1/3 S_0. Because they have the same error term but the one from S_1 (2N steps)
    # is a factor 4 smaller than the one from S_0 (N steps), we scale it with a factor 4.
    x_squared_trapz_2N = trapezoid(1, 5, x_squared, 2 * N)
    x_squared_trapz_combined = (4 / 3 * x_squared_trapz_2N) - (1 / 3 * x_squared_trapz)
    print(f"{x_squared_trapz_combined=}")

    sin_x_trapz_2N = trapezoid(0, pi, np.sin, 2 * N)
    sin_x_trapz_combined = (4 / 3 * sin_x_trapz_2N) - (1 / 3 * sin_x_trapz)
    print(f"{sin_x_trapz_combined=}")
    # Apparently there is a small difference between Simpson's rule and the 4/3S_1 - 1/3S_0 rule which goes to 0 for N goes to infinity

    # Assignment 1c: Romberg integration
    x_squared_romberg = romberg(1, 5, x_squared, N_start=N, order=6)
    print(f"Romberg result: {x_squared_romberg=}")
    sin_x_romberg = romberg(0, pi, np.sin, N_start=N, order=6)
    print(f"Romberg result: {sin_x_romberg=}")

    x_squared_romberg_vector = romberg_vector_version(1, 5, x_squared, N_start=N, order=6)
    print(f"Romberg_vector result: {x_squared_romberg_vector=}")
    sin_x_romberg_vector = romberg_vector_version(0, pi, np.sin, N_start=N, order=6)
    print(f"Romberg_vector result: {sin_x_romberg_vector=}")

    # Assignment 1d
    a, b = 0, 10
    for n in [10, 100, 1000, 10000]:
        q1d_func_trapz = trapezoid(a, b, q1d_func, n)
        print(f"{n=}, {q1d_func_trapz=}")
        q1d_func_simp_vec = simpson_vectorized(a, b, q1d_func, n)
        print(f"{n=}, {q1d_func_simp_vec=}")

    for m in [2, 4, 6]:
        q1d_func_romberg = romberg(a, b, q1d_func, N_start=10, order=m)
        print(f"{m=}, {q1d_func_romberg=}")
        q1d_func_romberg_vec = romberg_vector_version(a, b, q1d_func, N_start=10, order=m)
        print(f"{m=}, {q1d_func_romberg_vec=}")

    # simpson is the most accurate, followed by trapezoid, after 10000 intervals
    # But Romberg is already closer with m (order) = 6, at (so 2^m)=64 intervals, than trapz and simpson are at 1000
    # Wolfram alpha calculation: integral_0^10 (3 exp(-2 x) + (x/10)^4) dx = 7/2 - 3/(2 e^20)≈3.5000


main()
