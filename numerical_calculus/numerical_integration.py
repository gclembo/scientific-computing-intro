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
    :return: Approximate definite integral using left-handed Riemann sum.
    """


def right_hand_sum(f: callable(float), a: float, b: float, steps: int) -> float:
    """
    Calculates and returns the approximate definite integral using
    right-handed Riemann sum for the given function f over a given
    interval from a to b with the given number of steps.
    :param f: Function to approximate integral using right-handed Riemann sum.
    :param a: Left point of interval to sum over.
    :param b: Right point of interval to sum over.
    :param steps: Number of steps to break interval into when performing the Riemann sum.
    :return: Approximate definite integral using right-handed Riemann sum.
    """


def midpoint_sum(f: callable(float), a: float, b: float, steps: int) -> float:
    """
    Calculates and returns the approximate definite integral using
    the midpoint Riemann sum for the given function f over a given
    interval from a to b with the given number of steps.
    :param f: Function to approximate integral using the midpoint Riemann sum.
    :param a: Left point of interval to sum over.
    :param b: Right point of interval to sum over.
    :param steps: Number of steps to break interval into when performing the Riemann sum.
    :return: Approximate definite integral using midpoint Riemann sum.
    """


def trapezoid_sum(f: callable(float), a: float, b: float, steps: int) -> float:
    """
    Calculates and returns the approximate definite integral using
    the trapezoid sum method for the given function f over a given
    interval from a to b with the given number of steps.
    :param f: Function to approximate integral using the trapezoid sum method.
    :param a: Left point of interval to sum over.
    :param b: Right point of interval to sum over.
    :param steps: Number of steps to break interval into when performing the trapezoid sum.
    :return: Approximate definite integral using the trapezoid sum method.
    """


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
    :raises ValueError: If the given number of steps is not positive or even.
    :return: Approximate definite integral using Simpson’s 1/3 rule.
    """
