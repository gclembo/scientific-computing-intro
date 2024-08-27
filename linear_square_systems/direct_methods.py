import numpy as np


def back_substitution(upper: np.array, b: np.array) -> np.array:
    """
    Given an upper triangular matrix a vector to augment to solve the system, returns the
    solution vector. If a free variable is encountered, 0 is the value used.

    :param upper: Square upper triangular matrix for a linear system.
    :param b: vector to augment and use for solving the system.
    :return: solution vector to system

    :raises ValueError: If the given upper triangular matrix is not square, if the upper triangular
    matrix and the vector to augment have a different number of rows, if the given upper triangular
    matrix is not upper triangular, or if there are no solutions to the system.
    """
    if upper.shape[0] != upper.shape[1]:
        raise ValueError("System is not square")
    if upper.shape[0] != b.shape[0]:
        raise ValueError("Both inputs must have same number of rows")
    if not np.array_equal(upper, np.triu(upper)):
        raise ValueError("System is not upper triangular")

    m = upper.shape[0]
    augmented = np.column_stack([upper, b])
    sol = np.array([None] * m).astype(np.float32)
    for i in range(m - 1, -1, -1):
        total = 0
        for j in range(m - 1, i, -1):
            total += augmented[i, j] * sol[j]

        x = 0  # Default value
        if augmented[i, i] == 0:
            if total == augmented[i, m]:
                print("x" + str(i) + " is free parameter. Using 0 as default.")
            else:
                raise ValueError("No solutions")
        else:
            x = (augmented[i, m] - total) / augmented[i, i]
        sol[i] = float(x)
    return sol.transpose()


# Given a regular matrix, performs regular gaussian elimination. Raises ValueError
#  if given matrix is not regular
def regular_gaussian_elim(input_matrix: np.array) -> np.array:
    """
    Given a regular matrix, performs regular gaussian elimination and returns the reduced matrix.
    :param input_matrix: A regular matrix to reduce using regular gaussian elimination.
    :return: Reduced matrix
    :raises ValueError: If the given matrix is not regular
    """
    m = input_matrix.shape[0]
    reduced = input_matrix.astype(np.float32)
    for i in range(m - 1):
        if reduced[i][i] == 0:
            raise ValueError("Given matrix is not regular")

        for j in range(i + 1, m):
            coef = reduced[j][i] / reduced[i][i]
            reduced[j] = reduced[j] - (coef * reduced[i].astype(np.float32))
    return reduced


def complete_gaussian_elim(input_matrix: np.array) -> np.array:
    """
    Given a matrix, performs complete Gaussian elimination and returns reduced matrix.
    :param input_matrix: Matrix to reduce with complete Gaussian elimination.
    :return: Reduced matrix.
    """
    m, n = input_matrix.shape
    reduced = input_matrix.astype(np.float32)
    curr_row = 0
    curr_col = 0
    while curr_row < m and curr_col < n:
        swap_row = curr_row + 1
        while reduced[curr_row][curr_col] == 0 and swap_row < m:
            reduced[[curr_row, swap_row]] = reduced[[swap_row, curr_row]]
            swap_row += 1

        if swap_row == m:
            curr_col += 1
            continue

        for j in range(curr_row + 1, m):
            coef = reduced[j][curr_col] / reduced[curr_row][curr_col]
            reduced[j] = reduced[j] - coef * reduced[curr_row]
        curr_row += 1
        curr_col += 1
    return reduced


# Given a regular square matrix, decomposes the matrix into lower and upper
#  triangular matrices and returns the lower and upper matrices.
def lu_decomposition(input_matrix: np.array) -> np.array:
    """
    Given a regular square matrix, decomposes the matrix into lower and upper triangular matrices
    and returns these matrices.
    :param input_matrix: Regular matrix to decompose.
    :returns: lower and upper matrices decomposed from the input matrix.
    :raises ValueError: If the given matrix is not square or if it is not regular.
    """
    if input_matrix.shape[0] != input_matrix.shape[1]:
        raise ValueError("Given matrix is not square")

    m = input_matrix.shape[0]
    l = np.eye(m)
    u = input_matrix.astype(np.float32)
    for i in range(m - 1):
        if u[i][i] == 0:
            raise ValueError("Given matrix is not regular")
        for j in range(i + 1, m):
            coef = u[j][i] / u[i][i]
            l[j][i] = coef
            u[j] = u[j] - coef * u[i]
    return l, u
