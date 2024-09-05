import numpy as np


def left_hand_sum(f: callable(float), a: float, b: float, steps: int) -> float:
    """
    Calculates and returns the approximate definite integral using
    left-handed Riemann sum for the given function f over a given
    interval from a to b with the given number of steps.
    :param f: Function to approximate integral using left-handed Riemann sum.
    :param a: Left point of interval to sum over.
    :param b: Right point of interval to sum over.
    :param steps: Number of steps to break interval into when performing the Riemann sum.
    :raises ValueError: If the given number of steps is not positive or even, or if the first
    endpoint is greater than the second endpoint for the given interval.
    :return: Approximate definite integral using left-handed Riemann sum.
    """
    if steps < 0:
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
    interval from a to b with the given number of steps.
    :param f: Function to approximate integral using right-handed Riemann sum.
    :param a: Left point of interval to sum over.
    :param b: Right point of interval to sum over.
    :param steps: Number of steps to break interval into when performing the Riemann sum.
    :raises ValueError: If the given number of steps is not positive or even, or if the first
    endpoint is greater than the second endpoint for the given interval.
    :return: Approximate definite integral using right-handed Riemann sum.
    """
    if steps < 0:
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
    interval from a to b with the given number of steps.
    :param f: Function to approximate integral using the midpoint Riemann sum.
    :param a: Left point of interval to sum over.
    :param b: Right point of interval to sum over.
    :param steps: Number of steps to break interval into when performing the Riemann sum.
    :raises ValueError: If the given number of steps is not positive or even, or if the first
    endpoint is greater than the second endpoint for the given interval.
    :return: Approximate definite integral using midpoint Riemann sum.
    """
    if steps < 0:
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
    interval from a to b with the given number of steps.
    :param f: Function to approximate integral using the trapezoid sum method.
    :param a: Left point of interval to sum over.
    :param b: Right point of interval to sum over.
    :param steps: Number of steps to break interval into when performing the trapezoid sum.
    :raises ValueError: If the given number of steps is not positive or even, or if the first
    endpoint is greater than the second endpoint for the given interval.
    :return: Approximate definite integral using the trapezoid sum method.
    """
    if steps < 0:
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
    interval from a to b with the given number of steps. The given
    number of intervals must be positive and even or a ValueError will be raised.
    :param f: Function to approximate integral using Simpson’s 1/3 rule.
    :param a: Left point of interval to sum over.
    :param b: Right point of interval to sum over.
    :param steps: Number of steps to break interval into when performing Simpson’s 1/3 rule.
    :raises ValueError: If the given number of steps is not positive or even, or if the first
    endpoint is greater than the second endpoint for the given interval.
    :return: Approximate definite integral using Simpson’s 1/3 rule.
    """
    if steps < 0:
        raise ValueError("Number of steps must be positive")
    if a >= b:
        raise ValueError("First endpoint must be less than second endpoint for the given interval")
    if steps % 2 == 1:
        raise ValueError("Number of steps for Simpson\'s Rule must be even")
    dx = (b - a) / steps
    x = np.linspace(a, b, steps + 1)
    y = f(x)
    return dx * (y[0] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]) + y[-1]) / 3
