import numpy as np


def bisection_method(f: callable(float), a: float, b: float,
                     threshold: float = 1e-5, iterations: int = 50) -> float:
    """
    Given a function, and two endpoints, finds the root between the two endpoints using the
    bisection method assuming the function is continuous, f(a) and f(b) have opposite signs,
    and there is exactly one root in between a and b. The search will iterate until
    successive approximations of x are less than 1e-5 or until 50 iterations are reached.
    These stopping criteria values can be specified by the user.

    Parameters
    ----------
    f : callable(float)
        Function to find root of.
    a : float
        Left x value of interval to search.
    b : float
        Right x value of interval to search.
    threshold : float, default 1e-5
        Minimum distance between successive estimations for x until the
        algorythm stops iterating. Defaults to 1e-5.
    iterations : int, default 50
        Number of iterations until algorythm stops iterating. Defaults to 50.
    Returns
    -------
    float
        Approximate x value of the root in the interval.

    Raises
    ------
    ValueError
        If the first endpoint is not less than the second endpoint or if the
        function evaluated at the endpoints don't have different signs.
    """
    if a >= b:
        raise ValueError("First endpoint must be less than second endpoint")
    f_a = f(a)
    f_b = f(b)

    if np.sign(f_a) == np.sign(f_b):
        raise ValueError("The function evaluated at the given endpoints must have different signs")
    if f_a == 0:
        return a
    if f_b == 0:
        return b
    iteration = 0
    dist = (b - a) / 2
    while iteration < iterations and threshold < dist:
        mid = (b + a) / 2
        f_mid = f(mid)
        if np.sign(f_a) == np.sign(f_mid):
            a = mid
            f_a = f_mid
        elif np.sign(f_mid) == np.sign(f_b):
            b = mid
            f_b = f_mid
        else:
            return mid
        iteration += 1
        dist /= 2
    return (a + b) / 2


def newton_method(f: callable(float), df: callable(float), x0: float,
                  threshold: float = 1e-5, iterations: int = 30) -> float:
    """
    Given a function, it's derivative, and an initial guess, attempts to find a root using
    Newton's Method assuming the function is differentiable, has a root, and that points
    near the root do not have a derivative of 0. The search will iterate until successive
    approximations of x are less than 1e-5 or until 30 iterations are reached.
    These stopping criteria values can be specified by the user.

    Parameters
    ----------
    f : callable(float)
        Function to find root of.
    df : callable(float)
        Derivative of function to find root of.
    x0 : float
        Initial guess for x value of root.
    threshold : float, default 1e-5
        Minimum distance between successive estimations for x until the
        algorythm stops iterating. Defaults to 1e-5.
    iterations : int, default 30
        Number of iterations until algorythm stops iterating. Defaults to 30.

    Returns
    -------
    float
        Approximate x value of root in the interval.

    Raises
    ------
    ValueError
        If the derivative at the initial guess is 0.

    """
    if df(x0) == 0:
        raise ValueError("Starting point must not have a derivative of 0")
    x1 = x0 - f(x0) / df(x0)
    iteration = 1
    dist = np.abs(x1 - x0)
    while iteration < iterations and threshold < dist:
        x0 = x1
        x1 = x0 - f(x0) / df(x0)
        iteration += 1
        dist = np.abs(x1 - x0)
    return x1


def secant_method(f: callable(float), x0: float, x1: float,
                  threshold: float = 1e-5, iterations: int = 30) -> float:
    """
    Given a function, and two initial guesses attempts to find a root using the Secant Method
    assuming the function has a root. The search will iterate until successive approximations
    of x are less than 1e-5 or until 30 iterations are reached.
    These stopping criteria values can be specified by the user.

    Parameters
    ----------
    f : callable(float)
        Function to find root of.
    x0 : float
        First initial guess for x value of root.
    x1 : float
        Second initial guess for x value of root.
    threshold : float
        Minimum distance between successive estimations for x until the
        algorythm stops iterating. Defaults to 1e-5.
    iterations : int
        Number of iterations until algorythm stops iterating. Defaults to 30.

    Returns
    -------
    float
        Approximate x value of root in the interval.

    Raises
    ------
    ValueError
        If the first two initial guesses evaluate by f to the same value.
    """
    if f(x0) == f(x1):
        raise ValueError("The function evaluated at the given points must have different values")
    x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
    iteration = 1
    dist = np.abs(x2 - x1)
    while iteration < iterations and threshold < dist:
        x0 = x1
        x1 = x2
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        iteration += 1
        dist = np.abs(x2 - x1)
    return x2
