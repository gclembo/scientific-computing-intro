import numpy as np


def back_substitution(upper: np.array, b: np.array) -> np.array:
    """
    Given an upper triangular matrix, a vector of dependent values to solve the system, returns the
    solution vector for the system. If a free variable is encountered, 0 is the value used.

    Parameters
    ----------
    upper : np.array
        Square upper triangular matrix for a linear system.
    b : np.array
        Vector of dependent values to solve the system.

    Returns
    -------
    np.array
        Solution vector for the system.

    Raises
    ------
    ValueError
        If the given upper triangular matrix is not square, if the upper
        triangular matrix and the vector of dependent values have a different number of rows,
        if the given matrix is not upper triangular, or if there are no solutions to the system.
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


def regular_gaussian_elim(input_matrix: np.array) -> np.array:
    """
    Given a regular matrix, performs regular gaussian elimination and returns the reduced matrix.

    Parameters
    ----------
    input_matrix : np.array
        A regular matrix to reduce using regular gaussian elimination.

    Returns
    -------
    np.array
        Reduced matrix.

    Raises
    ------
    ValueError
        If the given matrix is not regular.
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

    Parameters
    ----------
    input_matrix : np.array
        Matrix to reduce with complete Gaussian elimination.

    Returns
    -------
    np.array
        Reduced matrix.
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


def lu_decomposition(input_matrix: np.array) -> np.array:
    """
    Given a regular square matrix, decomposes the matrix into lower and upper triangular matrices
    and returns these matrices.

    Parameters
    ----------
    input_matrix : np.array
        Regular matrix to decompose.

    Returns
    -------
    np.array
        Lower and upper matrices decomposed from the input matrix.

    Raises
    ------
    ValueError
        If the given matrix is not square or if it is not regular.
    """
    if input_matrix.shape[0] != input_matrix.shape[1]:
        raise ValueError("Given matrix is not square")

    m = input_matrix.shape[0]
    lower = np.eye(m)
    upper = input_matrix.astype(np.float32)
    for i in range(m - 1):
        if upper[i][i] == 0:
            raise ValueError("Given matrix is not regular")
        for j in range(i + 1, m):
            coef = upper[j][i] / upper[i][i]
            lower[j][i] = coef
            upper[j] = upper[j] - coef * upper[i]
    return lower, upper
