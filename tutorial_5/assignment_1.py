from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return x**2 * np.sin(x)


def func_derivative(x):
    return 2 * x * np.sin(x) + x**2 * np.cos(x)


# Assignment 1a
x = np.linspace(0, 2 * np.pi, num=200)
plt.figure()
plt.plot(x, func_derivative(x), label="analytical derivative")


def central_difference(func: Callable, h: float, x: np.ndarray):
    return func(x + h) - func(x - h) / (2 * h)


# Assignment 1b
# Use the central difference method to calculate the numerical derivative for different values of the step size h
# It turns out the one with the largest step size (h=1) resembles the analytical derivative the best
fig, axs = plt.subplots(1, 4)
for i, h in enumerate([1, 0.1, 0.01, 0.001]):
    axs[i].plot(x, central_difference(func, h, x))
    axs[i].set_title(f"{h=}")
plt.show()

# Assignment 1c
# Use Ridder's method (Richardson extrapolation for differentiation)
