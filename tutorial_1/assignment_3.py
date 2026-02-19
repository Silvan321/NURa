from tracemalloc import start
import numpy as np
from matplotlib.image import imread
from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt

image = imread(r"C:\Users\toetseb\Documents\NURa\tutorial_1\M42_128.jpg")
# imshow(image)
# show()
print(f"Image shape: {image.shape}")


class BaseInterpolater:
    def find_starting_index(self, x: float, xdata: np.ndarray, m: int) -> int:
        """X is the value to interpolate the function f at, aka we want to know f(x).
        xdata are the x values of the measured data points.
        m is the order of the interpolation, aka m=2 linear interpolation.
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
        return max(0, min(len(xdata) - m, j_low - ((m - 2) // 2)))

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
base_interpolator.find_starting_index(x_value, x_array, 2)


class LinearInterpolater(BaseInterpolater):
    def interpolate(self, x, xdata, ydata):
        starting_index = super().find_starting_index(x, xdata, 2)
        return ydata[starting_index] + ((x - xdata[starting_index]) / (xdata[starting_index + 1] - xdata[starting_index])) * (
            ydata[starting_index + 1] - ydata[starting_index]
        )


linear_interpolater = LinearInterpolater()
linear_interpolater.interpolate(x_value, x_array, y_array)

# Assignment 3a
first_row = np.asarray(image[0], dtype=np.float64)
interpolated_row_xdata = np.linspace(0, 128, num=201)
interpolated_row_ydata = np.zeros(shape=interpolated_row_xdata.shape)
first_row_indices = range(len(first_row))
for index, new_x in enumerate(interpolated_row_xdata):
    interpolated_pixel = linear_interpolater.interpolate(new_x, first_row_indices, first_row)
    interpolated_row_ydata[index] = interpolated_pixel

print(first_row)
print(interpolated_row_xdata)

plt.figure()
plt.scatter(first_row_indices, first_row, marker=".")
plt.scatter(interpolated_row_xdata, interpolated_row_ydata, linestyle="dashed", marker=",")
plt.show()
