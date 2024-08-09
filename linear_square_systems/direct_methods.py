import numpy as np


# Given a upper triangular augmented a vector , returns the solution vector if there is unique solution
def back_substitution(upper, b):
    m = upper.shape[0]
    augmented = np.column_stack([upper, b])
    sol = np.array([None] * m)
    for i in range(m - 1, -1, -1):
        total = 0
        for j in range(m - 1, i, -1):
            total += augmented[i, j] * sol[j]

        x = 1  # Default value
        if augmented[i, i] == 0:
            if total == augmented[i, m]:
                print("x" + str(i) + " is free parameter. Using 1 as default.")
            else:
                raise ValueError("No solutions")
        else:
            x = (augmented[i, m] - total ) / augmented[i, i]
        sol[i] = float(x)
    return sol.transpose()


# Given a regular augmented, performs regular gaussian elimination. Raises ValueError
#  if given augmented is not regular
def regular_gaussian_elim(input_matrix):
    m = input_matrix.shape[0]
    reduced = input_matrix.copy()
    for i in range(m - 1):
        if reduced[i][i] == 0:
            raise ValueError("Given augmented is not regular")

        for j in range(i + 1, m):
            coef = reduced[j][i] / reduced[i][i]
            reduced[j] = reduced[j] - coef * reduced[i]
    return reduced


def complete_gaussian_elim(input_matrix):
    m = input_matrix.shape[0]
    reduced = input_matrix.copy()
    for i in range(m - 1):
        row = i + 1
        while reduced[i][i] == 0 and row != m:
            reduced[[i, row]] = reduced[[row, i]]
            row += 1

        if reduced[i][i] == 0:
            continue

        for j in range(i + 1, m):
            coef = reduced[j][i] / reduced[i][i]
            reduced[j] = reduced[j] - coef * reduced[i]
    return reduced


# Given a regular square augmented, decomposes the augmented into lower and upper
#  triangular matrices
def lu_decomposition(input_matrix):
    m = input_matrix.shape[0]
    l = np.eye(m)
    u = input_matrix.copy()
    for i in range(m - 1):
        if u[i][i] == 0:
            raise ValueError("Given argumented is not regular")
        for j in range(i + 1, m):
            coef = u[j][i] / u[i][i]
            l[j][i] = coef
            u[j] = u[j] - coef * u[i]
    return l, u

