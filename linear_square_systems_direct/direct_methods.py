import numpy as np


def back_substitution(matrix: np.array) -> np.array:
    n = matrix.shape[0]
    sol = np.array([None] * n)
    for i in range(n - 1, -1, -1):
        total = 0
        for j in range(n - 1, i, -1):
            total += matrix[i, j] * sol[j]
        x = (matrix[i, n] / matrix[i, i]) - total
        sol[i] = float(x)
    return sol.transpose()


system = np.array([[1, 2, 3], [0, 3, 4]])
print(system)
solution = back_substitution(system)
print(solution)
