import numpy as np


# Given a upper triangular matrix, returns the solution vector if there is unique solution
def back_substitution(matrix):
    n = matrix.shape[0]
    sol = np.array([None] * n)
    for i in range(n - 1, -1, -1):
        total = 0
        for j in range(n - 1, i, -1):
            total += matrix[i, j] * sol[j]

        x = 1 # Default value
        if matrix[i, i] == 0:
            if total == matrix[i, n]:
                print("x" + str(i) + " is free parameter. Using 1 as default.")
            else:
                raise ValueError("No solutions")
        else:
            x = (matrix[i, n] / matrix[i, i]) - total
        sol[i] = float(x)
    return sol


# Given a regular matrix, performs regular gaussian elimination. Raises ValueError
#  if given matrix is not regular
def normal_gaussian_elim(in_matrix):
    n = in_matrix.shape[0]
    out_matrix = in_matrix.copy()
    for i in range(n - 1):
        if out_matrix[i][i] == 0:
            raise ValueError("Given matrix is not regular")

        for j in range(i + 1, n):
            coef = out_matrix[j][i] / out_matrix[i][i]
            out_matrix[j] = out_matrix[j] - coef * out_matrix[i]

    return out_matrix


system = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 0]])
print(system)
print(back_substitution(system))
