import numpy as np


def bisection_method(f: callable(float), a: float, b: float,
                     threshold: float = 1e-5, steps=50) -> float:
    """
    Given a function, and two endpoints, finds the root between the two endpoints using the
    bisection method assuming the function is continuous, f(a) and f(b) have oposite signs,
    and there is exactly one root in between a and b. The search will iterate until
    successive approximations of x are less than 1e-5 or until 50 iterations are reached.
    These stopping criteria values can be specified by the user.
    :param f: Function to find root of.
    :param a: Left x value of interval to search.
    :param b: Right x value of interval to search.
    :param threshold: Minimum distance between successive estimations for x until the
     algorythm stops iterating. Defaults to 1e-5
    :param steps: Number of steps until algorythm stops iterating. Defaults to 50.
    :return: Approximate x value of root in the interval.
    """


def newton_method(f: callable(float), df: callable(float), x0: float,
                  threshold: float = 1e-5, steps=30) -> float:
    """
    Given a function, it's derivative, and an initial guess attempts to find a root using
    Newton's Method assuming the function is differentiable, has a root, and that points
    near the root do not have a derivative of 0. The search will iterate until successive
    approximations of x are less than 1e-5 or until 30 iterations are reached.
    These stopping criteria values can be specified by the user.
    :param f: Function to find root of.
    :param df: Derivative of function to find root of.
    :param x0: Initial guess for x value of root.
    :param threshold: Minimum distance between successive estimations for x until the
     algorythm stops iterating. Defaults to 1e-5
    :param steps: Number of steps until algorythm stops iterating. Defaults to 50.
    :return: Approximate x value of root in the interval.
    """


def secant_method(f: callable(float), x0: float, x1: float,
                  threshold: float = 1e-5, steps=30) -> float:
    """
    Given a function, and two initial guesses attempts to find a root using the Secant Method
    assuming the function has a root. The search will iterate until successive approximations
    of x are less than 1e-5 or until 30 iterations are reached.
    These stopping criteria values can be specified by the user.
    :param f: Function to find root of.
    :param x0: First initial guess for x value of root.
    :param x1: Second initial guess for x value of root.
    :param threshold: Minimum distance between successive estimations for x until the
     algorythm stops iterating. Defaults to 1e-5
    :param steps: Number of steps until algorythm stops iterating. Defaults to 50.
    :return: Approximate x value of root in the interval.
    """