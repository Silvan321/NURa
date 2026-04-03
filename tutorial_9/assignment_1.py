# Robust estimation

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
from matplotlib import pyplot as plt

from tutorial_8.assignment_1 import LevenbergMarquardt

# Only needed once
# import wget

# for i in range(1, 6):
#     filename = wget.download(url=f"http://home.strw.leidenuniv.nl/~marinichenko/NUR_A/dataset{i}.txt")

# Fitting options:
# 1. Levenberg Marquardt minimize chi squared
# 2. Minimize model made with 2 params using e.g. downhill simplex
# 3. builtin np.polyfit: gives coefficients


def partial_a(x):
    return x


def partial_b(x):
    return 1


if __name__ == "__main__":
    # 1a: Load and plot all datasets
    dataset_to_study = 1
    this_directory = Path(__file__).parent
    for file in this_directory.iterdir():
        if file.is_file() and file.suffix == ".txt":
            data = np.loadtxt(this_directory / file.name)
            x_loaded = data[:, 0]
            y_loaded = data[:, 1]
            # plt.figure()
            # plt.scatter(x_loaded, y_loaded, marker=".")
            # plt.show()

    this_directory = Path(__file__).parent
    data = np.loadtxt(this_directory / "dataset1.txt")  # Choose dataset 1 to start with
    x = data[:, 0]
    y = data[:, 1]

    # 1b: linear fitting
    # Let us use Levenberg Marquardt for Least Squares fitting by setting sigma constant for all x's
    # A linear relation means we use the function f(x) = ax + b to fit the data
    # Then our parameter partial derivatives are partial_a = x and partial b = 1
    partial_derivative_list = [partial_a, partial_b]
    sigma = 1  # Estimate not too big otherwise outliers are not outliers anymore
    lm = LevenbergMarquardt(x, y, partial_derivative_list, sigma)
    print(lm.solve_delta_p())
