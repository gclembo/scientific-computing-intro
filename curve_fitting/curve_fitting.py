import numpy as np


def least_squares_sol(a: np.array, b: np.array) -> np.array:
    """
    Given a matrix a and a vector b, returns the least squares solution to ax = b.

    Parameters
    ----------
    a : np.array
        Coefficient matrix.
    b : np.array
        Dependent values.

    Returns
    -------
    np.array
        Least squares solution x to ax = b.
    """
    normal_mat = np.matmul(a.transpose(), a)
    normal_b = np.matmul(a.transpose(), b)
    return np.linalg.solve(normal_mat, normal_b)


def poly_fit(x: np.array or list[float], y: np.array or list[float], n: int) -> np.array:
    """
    Given a vector of x values, a vector of y values, and an integer n, returns the
    coefficients for an nth degree polynomial in increasing order of degree
    using least squares error.

    Parameters
    ----------
    x : np.array or list[float]
        x values to fit curve to.
    y : np.array or list[float]
        y values to fit curve to.
    n : int
        Degree of polynomial.

    Returns
    -------
    np.array
        Vector of polynomial coefficients in increasing order of degree.
    """
    vandermonde_mat = np.empty((len(x), n + 1))
    for i in range(len(x)):
        for j in range(n + 1):
            vandermonde_mat[i][j] = x[i] ** j
    return least_squares_sol(vandermonde_mat, y)


def func_fit(x: np.array or list[float], y: np.array or list[float],
             functions: list[callable(float)]) -> np.array:
    """
    Given a vector of x values, a vector of y values, and a vector of mathematical functions
    to define a curve by a linear combination of these functions, returns a vector
    containing the constant term followed by the coefficients for each function after the
    curve is fitted using least squares error.

    Parameters
    ----------
    x : np.array or list[float]
        x values to fit curve to.
    y : np.array or list[float]
        y values to fit curve to.
    functions : list[callable(float)]
        Functions added together for curve to fit to data.

    Returns
    -------
    np.array
        Vector of constant term followed by function coefficients.
    """
    func_val_mat = np.concatenate([np.ones((len(x), 1)),
                                   np.empty((len(x), len(functions)))], axis=1)
    for i in range(len(x)):
        for j in range(len(functions)):
            func_val_mat[i][j + 1] = functions[j](x[i])
    return least_squares_sol(func_val_mat, y)
