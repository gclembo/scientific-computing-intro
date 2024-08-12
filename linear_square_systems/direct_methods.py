import numpy as np


# Given a upper triangular matrix a vector , returns the solution vector if there is unique solution
def back_substitution(upper: np.array, b: np.array) -> np.array:
    if upper.shape[0] != upper.shape[1]:
        raise ValueError("System is not square")
    if upper.shape[0] != b.shape[0]:
        raise ValueError("Both inputs must have same number of rows")
    if not np.array_equal(upper, np.triu(upper)):
        raise ValueError("System is not upper triangular")

    m = upper.shape[0]
    augmented = np.column_stack([upper, b]).astype(np.float32)
    sol = np.array([None] * m)
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
    m = input_matrix.shape[0]
    reduced = input_matrix.astype(np.float32)
    for i in range(m - 1):
        if reduced[i][i] == 0:
            raise ValueError("Given matrix is not regular")

        for j in range(i + 1, m):
            coef = reduced[j][i] / reduced[i][i]
            reduced[j] = reduced[j] - (coef * reduced[i].astype(np.float32))
    return reduced


# Given a matrix, performs regular gaussian elimination
def complete_gaussian_elim(input_matrix: np.array) -> np.array:
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
