from copy import deepcopy
from tracemalloc import start
import numpy as np
from matplotlib.image import imread
from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt

image = imread(r"C:\Users\toetseb\Documents\nura\NURa\tutorial_1\M42_128.jpg")
# imshow(image)
# show()
print(f"Image shape: {image.shape}")


class BaseInterpolater:
    def find_starting_index_and_closest_index(self, x: float, xdata: np.ndarray, m: int) -> tuple[int, int]:
        """X is the value to interpolate the function f at, aka we want to know f(x).
        xdata are the x values of the measured data points.
        m is the order of the interpolation, aka m=2 linear interpolation.
        we want to find the starting index of the points we want to use for interpolation and the index of the values in xdata closest to x.
        """
        if not self._test_xdata_monotonic(xdata):
            raise ValueError("xdata should be monotonic")
        j_low = 0  # lowest index
        j_high = len(xdata) - 1  # highest index
        while (j_high - j_low) > 1:
            j_middle = (j_high + j_low) // 2
            if x >= xdata[j_middle]:
                j_low = j_middle
            else:
                j_high = j_middle
        return max(
            0, min(len(xdata) - m, j_low - ((m - 2) // 2))
        ), j_low  # j_low now holds the midpoint. the higher the order m of interpolation, the more we have to go back to find the starting index.

    # -2 because when m=2, the middle point is the starting index. Because a higher order polynomial uses points on either side of x, we divide m by 2 when looking for the starting index.

    def _test_xdata_monotonic(self, xdata) -> bool:
        if xdata[1] > xdata[0]:  # xdata should be monotonically increasing
            return self._test_xdata_monotonically_increasing(xdata)
        if xdata[1] < xdata[0]:  # xdata should be monotonically decreasing
            return self._test_xdata_monotonically_decreasing(xdata)
        return False  # Catch case where first two values are equal

    def _test_xdata_monotonically_increasing(self, xdata) -> bool:
        return all(xdata[i] > xdata[i - 1] for i in range(1, len(xdata)))

    def _test_xdata_monotonically_decreasing(self, xdata) -> bool:
        return all(xdata[i] < xdata[i - 1] for i in range(1, len(xdata)))


x_value = 4.5
x_array = np.array((1, 2, 3, 4, 5, 6, 7))
y_array = np.array((4, 3, 2, 1, 2, 3, 4))
base_interpolator = BaseInterpolater()
base_interpolator.find_starting_index_and_closest_index(x_value, x_array, 2)


class LinearInterpolater(BaseInterpolater):
    def interpolate(self, x, xdata, ydata):
        starting_index, _ = super().find_starting_index_and_closest_index(x, xdata, 2)
        return ydata[starting_index] + ((x - xdata[starting_index]) / (xdata[starting_index + 1] - xdata[starting_index])) * (ydata[starting_index + 1] - ydata[starting_index])


class PolynomialInterpolater(BaseInterpolater):
    """Class to do polynomial interpolation using Neville's algorithm.
    m is the order of the polynomial.
    """

    def interpolate(self, x, xdata, ydata, m):
        starting_index, closest_index = super().find_starting_index_and_closest_index(x, xdata, m)
        interpolation_points = xdata[starting_index : starting_index + m - 1]
        P = deepcopy(ydata[starting_index : starting_index + m - 1])  # vector holding the polynomials
        initial_solution = ydata[closest_index]
        for k in range(1, m - 1):
            for i in range(m - 1 - k):
                j = i + k
                P[i] = ((xdata[j] - x) * P[i] + (x - xdata[i]) * P[j]) / (xdata[j] - xdata[i])  # e.g. (x_1-x)P_0 + (x-x_0)P_1 / x_1-x_0
            if k == (m - 2):  # last addition, get ready to save the error estimate
                self.error_estimate = np.abs(P[0] - P[1])
        interpolated_value_at_x = P[0]
        return interpolated_value_at_x


linear_interpolater = LinearInterpolater()
linear_interpolater.interpolate(x_value, x_array, y_array)

# Assignment 3a
first_row = np.asarray(image[0], dtype=np.float64)
interpolated_row_xdata = np.linspace(0, 128, num=201)
linear_interpolated_row_ydata = np.zeros(shape=interpolated_row_xdata.shape)
polynomial_interpolated_row_ydata = np.zeros(shape=interpolated_row_xdata.shape)

first_row_indices = range(len(first_row))
for index, new_x in enumerate(interpolated_row_xdata):
    linear_interpolated_pixel = linear_interpolater.interpolate(new_x, first_row_indices, first_row)
    linear_interpolated_row_ydata[index] = linear_interpolated_pixel

    polynomial_interpolated_pixel = PolynomialInterpolater().interpolate(new_x, first_row_indices, first_row, m=4)
    polynomial_interpolated_row_ydata[index] = polynomial_interpolated_pixel

print(first_row)
print(interpolated_row_xdata)

plt.figure(figsize=[10, 6])
plt.scatter(first_row_indices, first_row, marker=".", label="data")
plt.scatter(interpolated_row_xdata, linear_interpolated_row_ydata, linestyle="dashed", marker=",", s=1, label="linear interpolation")
plt.scatter(interpolated_row_xdata, polynomial_interpolated_row_ydata, linestyle="dashed", marker="*", s=1, label="polynomial interpolation")
plt.legend()
plt.ylim(-100, 100)
plt.show()
