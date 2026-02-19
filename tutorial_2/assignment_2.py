import numpy as np
from copy import deepcopy


class GaussJordan:
    def solve(self, a: np.ndarray, b: np.ndarray, inverse: bool = True) -> np.ndarray | None:
        """Implementation of the Gauss Jordan algorithm. Based on Numerical Recipes.
        Note that this implementation only works for a single b vector.
        The book implementation works for multiple b's at once.
        the bool argument indicates if we want the inverse to be calculated
        """
        self.ncolumns, self.nrows = a.shape[1], a.shape[0]
        self.a_inverse = np.identity(self.ncolumns)  # a should be a square matrix
        for column_index in range(self.ncolumns):
            pivot: float = 0.0
            pivot_row_index: int = 0
            for row_index in range(self.nrows):
                if (current_absolute_value := np.abs(a[row_index][column_index])) > pivot:
                    pivot = a[row_index][column_index]
                    pivot_row_index = row_index
            if pivot_row_index != column_index:  # if pivot is already in the correct (diagonal) position, we don't need to swap
                self._swap_row(
                    a, b, column_index, pivot_row_index
                )  # Note that we use the COLUMN index, as the pivot element should end up at the diagonal position, i.e. where column_index == row_index
                if inverse:
                    self._swap_inverse_rows(column_index, pivot_row_index)
            pivot_row_index = column_index
            if pivot == 0:
                raise ValueError(
                    "Error: Singular matrix"
                )  # this only happens when all elements in that column are zero. If that happens, two equations are degenerate (i.e. represent same info) and matrix is singular
            # First divide all elements in the pivot row by the pivot element. Optimize by only calculating inverse of pivot once
            pivot_inverse = 1 / pivot
            for pivot_column_index in range(self.ncolumns):
                a[pivot_row_index][pivot_column_index] *= pivot_inverse
                self.a_inverse[pivot_row_index][pivot_column_index] *= pivot_inverse
            b[pivot_row_index] *= pivot_inverse
            a[pivot_row_index][pivot_row_index] = 1.0
            # Now do the subtraction of all the other rows, including the b elements
            self._reduce_rows(a, b, column_index, pivot_row_index, inverse)
        if inverse:
            return self.a_inverse
        else:
            return None

    def _swap_row(self, a, b, row_index_1: int, row_index_2: int):
        temp_row = deepcopy(a[row_index_1])
        a[row_index_1] = a[row_index_2]
        a[row_index_2] = temp_row
        b[row_index_1], b[row_index_2] = b[row_index_2], b[row_index_1]  # Don't forget to swap the (scalar) b's!

    def _swap_inverse_rows(self, row_index_1: int, row_index_2: int):
        temp_inv_row = deepcopy(self.a_inverse[row_index_1])
        self.a_inverse[row_index_1] = self.a_inverse[row_index_2]
        self.a_inverse[row_index_2] = temp_inv_row

    def _reduce_rows(self, a: np.ndarray, b: np.ndarray, starting_column_index: int, pivot_row_index: int, inverse: bool):
        """Reduce all rows with the current pivot row. Notice that we only need to do so for columns that are to the right of the current pivot column,
        as everything to the right of the pivot column has already been reduced. Note that we do not even use the pivot value itself anymore at this step!
        It is not supplied as an argument to this private function.
        """
        if inverse:
            starting_column_index = 0
        for row_index in range(self.nrows):  # Reduce the rows, except for the pivot row
            if row_index != pivot_row_index:
                coefficient_in_pivot_column_of_row_to_be_reduced = a[row_index][pivot_row_index]  # Remember pivot_row_index == pivot_column_index because we placed it on the diagonal!
                for inner_column_index in range(starting_column_index, self.ncolumns):  # IF WE WANT THE INVERSE, WE CANNOT SKIP THE COLUMNS ON THE LEFT HAND SIDE!
                    a[row_index][inner_column_index] -= coefficient_in_pivot_column_of_row_to_be_reduced * a[pivot_row_index][inner_column_index]
                    if inverse:
                        self.a_inverse[row_index][inner_column_index] -= coefficient_in_pivot_column_of_row_to_be_reduced * self.a_inverse[pivot_row_index][inner_column_index]
                b[row_index] -= coefficient_in_pivot_column_of_row_to_be_reduced * b[pivot_row_index]


def main():
    A = np.array(
        [
            [3, 8, 1, -12, -4],
            [1, 0, 0, -1, 0],
            [4, 4, 3, -40, -3],
            [0, 2, 1, -3, -2],
            [0, 1, 0, -12, 0],
        ],
        dtype=float,
    )
    b = np.array([2, 0, 1, 0, 0], dtype=float)

    A_copy = deepcopy(A)
    b_copy = deepcopy(b)

    gauss_jordan_solver = GaussJordan()
    A_inverse = gauss_jordan_solver.solve(A, b)
    print(f"A, which is now the identity matrix: {A=}, \nb which now holds x: {b=}, \nA_inverse, which started of as the identity matrix: {A_inverse=}")
    # np.allclose(np.dot(A_copy, A_inverse), np.identity)

    # Verify using np.linalg.solve
    x_numpy = np.linalg.solve(A_copy, b_copy)
    a_inverse_numpy = np.linalg.inv(A_copy)
    print(f"{x_numpy=}")
    print(f"{a_inverse_numpy}")
    np.allclose(np.dot(A_copy, x_numpy), b_copy)


main()
