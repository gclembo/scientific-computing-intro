import numpy as np
from scipy.optimize import fsolve


def forward_approximation(f: callable(float), x: float, h: float) -> float:
    """
    Given a function, an x value, and a step size, approximates the derivative
    at the given point using the forward difference approximation with a step size of h.
    :param f: Function to approximate derivative of.
    :param x: Value to approximate derivative of f at.
    :param h: Step size in forward difference approximation.
    :return: Approximate derivative of f at x.
    """
    return (f(x + h) - f(x)) / h


def backward_approximation(f: callable(float), x: float, h: float) -> float:
    """
    Given a function, an x value, and a step size, approximates the derivative
    at the given point using the backward difference approximation with a step size of h.
    :param f: Function to approximate derivative of.
    :param x: Value to approximate derivative of f at.
    :param h: Step size in backward difference approximation.
    :return: Approximate derivative of f at x.
    """
    return (f(x) - f(x - h)) / h


def central_approximation(f: callable(float), x: float, h: float) -> float:
    """
    Given a function, an x value, and a step size, approximates the derivative
    at the given point using the central difference approximation with a step size of h.
    :param f: Function to approximate derivative of.
    :param x: Value to approximate derivative of f at.
    :param h: Step size in central difference approximation.
    :return: Approximate derivative of f at x.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def forward_euler(dy: callable(float), y0: float, t0: float, t: float, n: int) -> float:
    """
    Given a function of t and y for the derivative of y with respect to t, an initial y value, an initial
    t value, a final t value, and the number of steps, uses the forward Euler approximation
    to approximate the value of y at the final time.
    :param dy: Function of t and y for the derivative of y with respect to t.
    :param y0: Initial y value.
    :param t0: Initial t value.
    :param t: Final t value.
    :param n: Number of steps.
    :return: Approximate value of y at the final t.
    """
    h = (t - t0) / n
    y1 = y0 + h * dy(t0, y0)
    for i in range(n - 1):
        t0 += h
        y0 = y1
        y1 = y0 + h * dy(t0, y0)
    return y1


def backward_euler(dy: callable(float), y0: float, t0: float, t: float, n: int) -> float:
    """
    Given a function of t and y for the derivative of y with respect to t, an initial y value, an initial
    t value, a final t value, and the number of steps, uses the backward Euler approximation
    to approximate the value of y at the final time.
    :param dy: Function of t and y for the derivative of y with respect to t.
    :param y0: Initial y value.
    :param t0: Initial t value.
    :param t: Final t value.
    :param n: Number of steps.
    :return: Approximate value of y at the final t.
    """
    h = (t - t0) / n
    y1 = fsolve(lambda x: y0 + h * dy(t0 + h, x) - x, np.array(y0))[0]

    for i in range(n - 1):
        t0 += h
        y0 = y1
        y1 = fsolve(lambda x: y0 + h * dy(t0 + h, x) - x, np.array(y0))[0]
    return float(y1)


def trapezoid_method(dy: callable(float), y0: float, t0: float, t: float, n: int) -> float:
    """
    Given a function of t and y for the derivative of y with respect to t,
    an initial y value, an initial t value, a final t value, and the number
    of steps, uses the trapezoid method approximation
    to approximate the value of y at the final time.
    :param dy: Function of t and y for the derivative of y with respect to t.
    :param y0: Initial y value.
    :param t0: Initial t value.
    :param t: Final t value.
    :param n: Number of steps.
    :return: Approximate value of y at the final t.
    """
    h = (t - t0) / n
    y_fore = y0 + h * dy(t0, y0)
    y_back = fsolve(lambda x: y0 + h * dy(t0 + h, x) - x, np.array(y0))[0]
    y1 = (y_fore + y_back) / 2
    for i in range(n - 1):
        t0 += h
        y0 = y1
        y_fore = y0 + h * dy(t0, y0)
        y_back = fsolve(lambda x: y0 + h * dy(t0 + h, x) - x, np.array(y0))[0]
        y1 = (y_fore + y_back) / 2
    return y1


def rk4(dy: callable(float), y0: float, t0: float, t: float, n: int) -> float:
    """
    Given a function of t and y for the derivative of y with respect to t, an initial y value,
    an initial t value, a final t value, and the number of steps, uses the Runge-Kutta 4
    approximation to approximate the value of y at the final time.
    :param dy: Function of t and y for the derivative of y with respect to t.
    :param y0: Initial y value.
    :param t0: Initial t value.
    :param t: Final t value.
    :param n: Number of steps.
    :return: Approximate value of y at the final t.
    """
    h = (t - t0) / n
    k1 = dy(t0, y0)
    k2 = dy(t0 + h / 2, y0 + (k1 * h) / 2)
    k3 = dy(t0 + h / 2, y0 + (k2 * h) / 2)
    k4 = dy(t0 + h, y0 + h * k3)
    y1 = y0 * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    for i in range(n - 1):
        t0 += h
        y0 = y1
        k1 = dy(t0, y0)
        k2 = dy(t0 + h / 2, y0 + (k1 * h) / 2)
        k3 = dy(t0 + h / 2, y0 + (k2 * h) / 2)
        k4 = dy(t0 + h, y0 + h * k3)
        y1 = y0 * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y1
