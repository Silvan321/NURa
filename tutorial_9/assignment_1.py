# Robust estimation

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
# Only needed once
# import wget

# for i in range(1, 6):
#     filename = wget.download(url=f"http://home.strw.leidenuniv.nl/~marinichenko/NUR_A/dataset{i}.txt")


this_directory = Path(__file__).parent
for file in this_directory.iterdir():
    if file.is_file() and file.suffix == ".txt":
        data = np.loadtxt(this_directory / file.name)
        x = data[:, 0]
        y = data[:, 1]
        plt.figure()
        plt.scatter(x, y, marker=".")
        plt.show()


# Fitting options:
# 1. Levenberg Marquardt minimize chi squared
# 2. Minimize model made with 2 params using e.g. downhill simplex
# 3. builtin np.polyfit: gives coefficients
