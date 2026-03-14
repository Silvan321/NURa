# Random Number Generators
# RNG "States" are based on all info that the RNG has available.
# The RNG should not reveal its entire info to the user, otherwise the user can predict the next number
# A and B are sub RNGs
# A output not based on B output, but uses intermediate output
# A and B both use their own seed


from math import pi
from matplotlib import pyplot as plt
import numpy as np


def theta_phi_1a(P1, P2):
    theta = pi * P1
    phi = 2 * pi * P2
    return theta, phi


def theta_phi_1b(P1, P2):
    theta = np.arccos(1 - 2 * P1)
    phi = 2 * pi * P2
    return theta, phi


def main():
    num_points = 1000
    # We need TWO DIFFERENT random number generators, otherwise every theta, phi pair uses the same random number
    # Then we get a helix with the shape theta = 2 phi
    P_u1 = np.random.uniform(0, 1, size=num_points)
    P_u2 = np.random.uniform(0, 1, size=num_points)
    r = np.ones(num_points)

    theta_1a, phi_1a = theta_phi_1a(P_u1, P_u2)
    x_1a = r * np.sin(theta_1a) * np.cos(phi_1a)
    y_1a = r * np.sin(theta_1a) * np.sin(phi_1a)
    z_1a = r * np.cos(theta_1a)

    radius_check = x_1a**2 + y_1a**2 + z_1a**2
    print(radius_check)

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(x_1a, y_1a, z_1a)

    theta_1b, phi_1b = theta_phi_1b(P_u1, P_u2)
    x_1b = r * np.sin(theta_1b) * np.cos(phi_1b)
    y_1b = r * np.sin(theta_1b) * np.sin(phi_1b)
    z_1b = r * np.cos(theta_1b)

    radius_check = x_1b**2 + y_1b**2 + z_1b**2
    print(radius_check)

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(x_1b, y_1b, z_1b)

    plt.show()
    # As we can see, the second option corrects for the fact that a surface element on a unit sphere is compressed near the poles
    # In the case of the theta phi generation in 1a, we get more values near the poles
    # Think of a globe: Meridian lines converge as you get closer to the poles


main()
