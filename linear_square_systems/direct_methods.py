import numpy as np


# Given a upper triangular matrix, returns the solution vector if there is unique solution
def back_substitution(matrix):  # TODO change to add b vector
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
def regular_gaussian_elim(in_matrix):
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
            coef = out_matrix[j][i] / out_matrix[i][i]  # TODO check for no sols
            out_matrix[j] = out_matrix[j] - coef * out_matrix[i]
    return out_matrix


# Given a regular square matrix, decomposes the matrix into lower and upper
#  triangular matrices
def lu_decomposition(in_matrix):
    m = in_matrix.shape[0]
    l = np.eye(m)
    u = in_matrix.copy()
    for i in range(m - 1):
        if u[i][i] == 0:
            raise ValueError("Given matrix is not regular")
        for j in range(i + 1, m):
            coef = u[j][i] / u[i][i]
            l[j][i] = coef
            u[j] = u[j] - coef * u[i]
    return l, u


system = np.array([[1, 2, 3], [2, 3, 4], [1, 1, 2]])
temp = lu_decomposition(system)
l = temp[0]
u = temp[1]
print(l)
print(u)
print(system)
print(np.matmul(l, u))
