import numpy as np


def left_hand_sum(f: callable(float), a: float, b: float, steps: int) -> float:
    """
    Calculates and returns the approximate definite integral using
    left-handed Riemann sum for the given function f over a given
    interval from a to b with the given number of steps. A ValueError will be raised
    if the given number of subintervals is not positive, or if the first
    endpoint is greater than or equal to the second endpoint for the given interval.

    Parameters
    ----------
    f : callable(float)
        Function to approximate integral using left-handed Riemann sum.
    a : float
        Left point of interval to sum over.
    b : float
        Right point of interval to sum over.
    steps : int
        Number of subintervals to break interval into when performing the Riemann sum.

    Returns
    -------
    float
        Approximate definite integral using left-handed Riemann sum.

    Raises
    ------
    ValueError
        If the given number of steps is not positive, or if the first
        endpoint is greater than or equal to the second endpoint for the given interval.
    """
    if steps <= 0:
        raise ValueError("Number of steps must be positive")
    if a >= b:
        raise ValueError("First endpoint must be less than second endpoint for the given interval")
    dx = (b - a) / steps
    x = np.linspace(a, b - dx, steps)
    y = f(x)
    return sum(y) * dx


def right_hand_sum(f: callable(float), a: float, b: float, steps: int) -> float:
    """
    Calculates and returns the approximate definite integral using
    right-handed Riemann sum for the given function f over a given
    interval from a to b with the given number of subintervals. A ValueError will be raised
    if the given number of subintervals is not positive, or if the first
    endpoint is greater than or equal to the second endpoint for the given interval.

    Parameters
    ----------
    f : callable(float)
        Function to approximate integral using right-handed Riemann sum.
    a : float
        Left point of interval to sum over.
    b : float
        Right point of interval to sum over.
    steps : int
        Number of subintervals to break interval into when performing the Riemann sum.

    Returns
    -------
    float
        Approximate definite integral using right-handed Riemann sum.

    Raises
    ------
    ValueError
        If the given number of steps is not positive, or if the first
        endpoint is greater than or equal to the second endpoint for the given interval.
    """
    if steps <= 0:
        raise ValueError("Number of steps must be positive")
    if a >= b:
        raise ValueError("First endpoint must be less than second endpoint for the given interval")
    dx = (b - a) / steps
    x = np.linspace(a + dx, b, steps)
    y = f(x)
    return sum(y) * dx


def midpoint_sum(f: callable(float), a: float, b: float, steps: int) -> float:
    """
    Calculates and returns the approximate definite integral using
    the midpoint Riemann sum for the given function f over a given
    interval from a to b with the given number of steps. A ValueError will be raised
    if the given number of subintervals is not positive, or if the first
    endpoint is greater than or equal to the second endpoint for the given interval.

    Parameters
    ----------
    f : callable(float)
        Function to approximate integral using the midpoint Riemann sum.
    a : float
        Left point of interval to sum over.
    b : float
        Right point of interval to sum over.
    steps : int
        Number of subintervals to break interval into when performing the Riemann sum.

    Returns
    -------
    float
        Approximate definite integral using midpoint Riemann sum.

    Raises
    ------
    ValueError
        If the given number of steps is not positive, or if the first
        endpoint is greater than or equal to the second endpoint for the given interval.
    """
    if steps <= 0:
        raise ValueError("Number of steps must be positive")
    if a >= b:
        raise ValueError("First endpoint must be less than second endpoint for the given interval")
    dx = (b - a) / steps
    x = np.linspace(a + dx / 2, b - dx / 2, steps)
    y = f(x)
    return sum(y) * dx


def trapezoid_sum(f: callable(float), a: float, b: float, steps: int) -> float:
    """
    Calculates and returns the approximate definite integral using
    the trapezoid sum method for the given function f over a given
    interval from a to b with the given number of steps. A ValueError will be raised
    if the given number of subintervals is not positive, or if the first
    endpoint is greater than or equal to the second endpoint for the given interval.

    Parameters
    ----------
    f : callable(float)
        Function to approximate integral using the trapezoid sum method.
    a : float
        Left point of interval to sum over.
    b : float
        Right point of interval to sum over.
    steps : int
        Number of subintervals to break interval into when performing the trapezoid sum.

    Returns
    -------
    float
        Approximate definite integral using the trapezoid sum method.

    Raises
    ------
    ValueError
        If the given number of steps is not positive, or if the first
        endpoint is greater than or equal to the second endpoint for the given interval.
    """
    if steps <= 0:
        raise ValueError("Number of steps must be positive")
    if a >= b:
        raise ValueError("First endpoint must be less than second endpoint for the given interval")
    dx = (b - a) / steps
    x = np.linspace(a, b, steps + 1)
    y = f(x)
    return dx * (sum(y[1:-1]) + sum(y)) / 2


def simpson_rule_sum(f: callable(float), a: float, b: float, steps: int) -> float:
    """
    Calculates and returns the approximate definite integral using
    the Simpson’s 1/3 rule for the given function f over a given
    interval from a to b with the given number of steps. A ValueError will be raised
    if the given number of subintervals is not even or not positive, or if the first
    endpoint is greater than or equal to the second endpoint for the given interval.

    Parameters
    ----------
    f : callable(float)
        Function to approximate integral using Simpson’s 1/3 rule.
    a : float
        Left point of interval to sum over.
    b : float
        Right point of interval to sum over.
    steps : int
        Number of subintervals to break interval into when performing Simpson’s 1/3 rule.

    Returns
    -------
    float
        Approximate definite integral using Simpson’s 1/3 rule.

    Raises
    ------
    ValueError
        If the given number of steps is not positive or even, or if the first
        endpoint is greater than or equal to the second endpoint for the given interval.
    """
    if steps <= 0:
        raise ValueError("Number of steps must be positive")
    if a >= b:
        raise ValueError("First endpoint must be less than second endpoint for the given interval")
    if steps % 2 == 1:
        raise ValueError("Number of iterations for Simpson\'s Rule must be even")
    dx = (b - a) / steps
    x = np.linspace(a, b, steps + 1)
    y = f(x)
    return dx * (y[0] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]) + y[-1]) / 3
