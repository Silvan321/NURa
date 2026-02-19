from timeit import timeit

import numpy as np
from astropy.constants import M_sun
from scipy.constants import G, c


def schwarzschild_radius(m):
    """Calculate the Schwarzschild radius in meters. Supply m in kg"""
    return (2 * G * m) / c**2


sun_rs = schwarzschild_radius(M_sun).value
print(f"{sun_rs=} meters")

# Assignment 2a
dist = np.random.default_rng().normal(loc=1e6 * M_sun.value, scale=1e5 * M_sun.value, size=10_000)

repeat_statement_number = 1000
time = timeit(lambda: [schwarzschild_radius(m) for m in dist], number=repeat_statement_number)
print(f"Time taken to calculate Schwarzschild radius for 10,000 masses {repeat_statement_number} times: {time:.4f} seconds")

cinv = 1 / c
cinv2 = cinv * cinv


def schwarzschild_radius_predefined_cinv2(m):
    """Calculate the Schwarzschild radius in meters. Supply m in kg"""
    return (2 * G * m) * cinv2


time = timeit(lambda: [schwarzschild_radius_predefined_cinv2(m) for m in dist], number=repeat_statement_number)
print(f"Time taken to calculate Schwarzschild radius for 10,000 masses with predefined cinv2 {repeat_statement_number} times: {time:.4f} seconds")
