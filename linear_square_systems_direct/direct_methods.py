import numpy as np


# Given a upper triangular matrix, returns the solution vector if there is unique solution
def back_substitution(matrix):
    m = matrix.shape[0]
    sol = np.array([None] * m)
    for i in range(m - 1, -1, -1):
        total = 0
        for j in range(m - 1, i, -1):
            total += matrix[i, j] * sol[j]

        x = 1  # Default value
        if matrix[i, i] == 0:
            if total == matrix[i, m]:
                print("x" + str(i) + " is free parameter. Using 1 as default.")
            else:
                raise ValueError("No solutions")
        else:
            x = (matrix[i, m] / matrix[i, i]) - total
        sol[i] = float(x)
    return sol


# Given a regular matrix, performs regular gaussian elimination. Raises ValueError
#  if given matrix is not regular
def normal_gaussian_elim(in_matrix):
    m = in_matrix.shape[0]
    out_matrix = in_matrix.copy()
    for i in range(m - 1):
        if out_matrix[i][i] == 0:
            raise ValueError("Given matrix is not regular")

        for j in range(i + 1, m):
            coef = out_matrix[j][i] / out_matrix[i][i]
            out_matrix[j] = out_matrix[j] - coef * out_matrix[i]
    return out_matrix


def complete_gaussian_elim(in_matrix):
    m = in_matrix.shape[0]
    out_matrix = in_matrix.copy()
    for i in range(m - 1):
        row = i + 1
        while out_matrix[i][i] == 0 and row != m:
            out_matrix[[i, row]] = out_matrix[[row, i]]
            row += 1

        if out_matrix[i][i] == 0:
            continue

        for j in range(i + 1, m):
            coef = out_matrix[j][i] / out_matrix[i][i]
            out_matrix[j] = out_matrix[j] - coef * out_matrix[i]
    return out_matrix


system = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [1, 0, 1, 0]])
print(system)
print(complete_gaussian_elim(system))
