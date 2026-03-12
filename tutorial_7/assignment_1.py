# Minimization and Maximization


from matplotlib import pyplot as plt
import numpy as np


def func1a(x):
    return x**4 + 10 * x**3 + 10 * (x - 2) ** 2


def main():
    x = np.linspace(-10, 10, num=100)

    plt.figure()
    plt.plot(x, func1a(x))
    plt.show()


main()
