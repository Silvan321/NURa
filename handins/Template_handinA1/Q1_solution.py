import numpy as np
from math import e
from scipy.stats import poisson
import math


def _factorial(k: np.int32):
    """Define the factorial function ourselves since we are not allowed to use the build it numpy function."""
    if not isinstance(k, np.int32):
        raise TypeError("k should be an np.int32!")
    if k < 0:
        raise ValueError("k should be a positive integer")
    product: np.int32 = np.int32(1)
    for integer in range(2, k + 1):  # Start at 2 since we don't want to multiply with 0 and 1, and stop at k+1 since k should be included
        product *= integer
    return product


def Poisson(k: np.int32, lmbda: np.float32) -> np.float32:
    """Calculate the Poisson probability for k occurrences with mean lmbda.
    Parameters:
        k (np.int32): The number of occurrences.
        lmbda (np.float32): The mean number of occurrences.
    Returns:
        np.float32: The probability of observing k occurrences given the mean lmbda.
    """
    print()
    print(k * np.log10(lmbda))
    print(lmbda * np.log10(e))
    print(sum(np.log10(i) for i in range(1, k)))

    def custom_poisson_pmf(k, lam):
        return lam**k * math.exp(-lam) / math.factorial(k)

    print(f"{poisson.pmf(int(k), float(lmbda))=}, {custom_poisson_pmf(int(k), float(lmbda))=}, {10 ** (k * np.log10(lmbda) - lmbda * np.log10(e) - sum(np.log10(i) for i in range(1, k+1)))=}")
    return 10 ** (k * np.log10(lmbda) - lmbda * np.log10(e) - sum(np.log10(i) for i in range(1, k)))


def main() -> None:
    # (lambda, k) pairs:
    values = [
        (np.float32(1.0), np.int32(0)),
        (np.float32(5.0), np.int32(10)),
        (np.float32(3.0), np.int32(21)),
        (np.float32(2.6), np.int32(40)),
        (np.float32(100.0), np.int32(5)),
        (np.float32(101.0), np.int32(200)),
    ]
    with open("Poisson_output.txt", "w") as file:
        for i, (lmbda, k) in enumerate(values):
            P = Poisson(k, lmbda)
            if i < len(values) - 1:
                file.write(f"{lmbda:.1f} & {k} & {P:.6e} \\\\ \\hline \n")
            else:
                file.write(f"{lmbda:.1f} & {k} & {P:.6e} \n")


if __name__ == "__main__":
    main()
