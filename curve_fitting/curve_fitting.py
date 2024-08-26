import numpy as np


def poly_fit(x: np.array or list[float], y: np.array or list[float], n: int) -> np.array:
    """
    Given a vector of x values, a vector of y values, and an integer n, returns the
    coefficients for an nth degree least squares polynomial.
    :param x: x values to fit curve to.
    :param y: y values to fit curve to.
    :param n: degree of polynomial.
    :return: vector of polynomial coefficients in increasing order of degree.
    """
    vandermonde_mat = np.empty((len(x), n + 1))
    for i in range(len(x)):
        for j in range(n + 1):
            vandermonde_mat[i][j] = x[i] ** j

    normal_mat = np.matmul(vandermonde_mat.transpose(), vandermonde_mat)
    normal_y = np.matmul(vandermonde_mat.transpose(), y)
    return np.linalg.solve(normal_mat, normal_y)


def func_fit(x: np.array or list[float], y: np.array or list[float],
             functions: list[callable(float)]) -> np.array:
    """
    Given a vector of x values, a vector of y values, and a vector of mathematical functions,
    to define a curve by adding together, returns a vector containing the constant term followed
    by the coefficients for each function after curve is fitted using least squares.
    :param x: x values to fit curve to.
    :param y: y values to fit curve to.
    :param functions: Functions added together for curve to fit to data.
    :return: Vector of constant term followed by function coefficients.
    """
    func_val_mat = np.concatenate([np.ones((len(x), 1)),
                                   np.empty((len(x), len(functions)))], axis=1)
    for i in range(len(x)):
        for j in range(len(functions)):
            func_val_mat[i][j + 1] = functions[j](x[i])

    normal_mat = np.matmul(func_val_mat.transpose(), func_val_mat)
    normal_y = np.matmul(func_val_mat.transpose(), y)
    return np.linalg.solve(normal_mat, normal_y)
