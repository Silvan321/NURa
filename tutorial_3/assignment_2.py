from copy import deepcopy

import numpy as np
import scipy


class GaussJordan:
    def solve(self, a: np.ndarray, b: np.ndarray, inverse: bool = True) -> np.ndarray | None:
        """Implementation of the Gauss Jordan algorithm. Based on Numerical Recipes.
        Note that this implementation only works for a single b vector.
        The book implementation works for multiple b's at once.
        the bool argument indicates if we want the inverse to be calculated
        """
        self.nrows, self.ncolumns = a.shape[0], a.shape[1]
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

    def _reduce_rows(self, a: np.ndarray, b: np.ndarray, starting_column_index: int, pivot_row_index: int, inverse: bool):  # noqa: FBT001
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


class LUDecomposition:
    def __init__(self, a: np.ndarray):
        """This class performs LU decomposition, where a matrix A is decomposed in two matrices L and U.
        Here L is a Lower Triangular matrix (only elements on or below diagonal), and U is an Upper triangular matrix (only elements on or above the diagonal).
        We can use these to solve the equation A*x = (L*U)*x = L*(U*x) = b, by first solving L*y=b and then U*x=y.
        Use the constructor of the class to do the decomposition, using Crout's algorithm (slide 12 lecture 3)
        Then the instance of the object holds the LU decomposed matrix.
        We can then call the solve method as many times as we like, for as many b's as we like.
        """
        self.nrows, self.ncolumns = a.shape[0], a.shape[1]
        self.LU_matrix = np.identity(self.ncolumns)  # step 1: construct an identity matrix to start
        for j in range(self.ncolumns):
            self.LU_matrix[0, j] = a[0, j]  # for i = 0, beta_0j = a_0j
            for i in range(1, self.nrows):  # starting from 0 or 1 should not make a difference, as the first row is already set to the correct values, but we can skip it to save some time
                if i <= j:
                    self.LU_matrix[i, j] = a[i, j] - sum(
                        self.LU_matrix[i, k] * self.LU_matrix[k, j] for k in range(i)
                    )  # beta_ij = a_ij - sum (alpha_ik beta_kj) from k=0 to i-1. range doesn't include end point!
                if i > j:
                    self.LU_matrix[i, j] = (a[i, j] - sum(self.LU_matrix[i, k] * self.LU_matrix[k, j] for k in range(j))) / self.LU_matrix[
                        j, j
                    ]  # alpha_ij = a_ij - sum (alpha_ik beta_kj) from k=0 to j-1. range doesn't include end point!

    def get_LU_decomposition(self):
        return self.LU_matrix

    def solve(self, b: np.ndarray) -> np.ndarray:
        # Forward substitution first: L*y=b
        b_size = len(b)
        y = np.zeros(b_size)
        x = np.zeros(b_size)
        y[0] = b[0]  # don't need to divide by LU_matrix[i,i] as for the forward substitution, the diagonal elements of L are all 1's
        for i in range(1, b_size):
            y[i] = b[i] - sum(self.LU_matrix[i, j] * y[j] for j in range(i))

        # Now do the backsubstitution
        x[b_size - 1] = y[b_size - 1] / self.LU_matrix[b_size - 1, b_size - 1]
        for i in reversed(range(b_size)):
            x[i] = (y[i] - sum(self.LU_matrix[i, j] * x[j] for j in range(i + 1, b_size))) / self.LU_matrix[i, i]
        return x


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

    A_copy_2 = deepcopy(A)
    b_copy_2 = deepcopy(b)

    gauss_jordan_solver = GaussJordan()
    A_inverse = gauss_jordan_solver.solve(A_copy, b_copy)
    print(f"A_copy, which is now the identity matrix: {A_copy=}, \nb_copy which now holds x: {b_copy=}, \nA_inverse, which started of as the identity matrix: {A_inverse=}")
    # np.allclose(np.dot(A_copy, A_inverse), np.identity)

    # Verify using np.linalg.solve
    x_numpy = np.linalg.solve(A_copy_2, b_copy_2)
    a_inverse_numpy = np.linalg.inv(A_copy_2)
    print(f"{x_numpy=}")
    print(f"{a_inverse_numpy}")
    np.allclose(np.dot(A_copy_2, x_numpy), b_copy_2)

    print(f"{A=}")
    lu_decomposition_instance = LUDecomposition(A)
    A_decomposed = lu_decomposition_instance.get_LU_decomposition()
    print(f"{A_decomposed=}")
    # verify if LU decomposed matrix is correct
    P, L_scipy, U_scipy = scipy.linalg.lu(A, permute_l=False)
    print(f"{L_scipy=}")
    print(f"{U_scipy=}")
    A_decomposed_scipy = P @ L_scipy + U_scipy - np.eye(A.shape[0])
    # A = L @ U so matrix multiplied. the decomposition of A is L + U - I as one has zeros where the other has nonzero elements and vice versa, except for the diagonal
    print(f"{A_decomposed_scipy=}")
    print(f"LU decomposition equal to scipy implementation: {np.allclose(A_decomposed, A_decomposed_scipy)}")

    x_lu = lu_decomposition_instance.solve(b)  # temp
    print(f"solution using LU decomposition equal to numpy: {np.allclose(x_lu, x_numpy)}")
    print(f"{x_lu=}")


main()
