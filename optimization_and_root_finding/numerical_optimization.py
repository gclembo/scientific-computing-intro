import numpy as np


def three_point_search(f: callable(float), x_min: float, x_max: float,
                       threshold: float = 1e-5, steps=50) -> float:
    """
    Given a function, a minimum x value, and a maximum x value, returns the approximate x value
    of the minimum value of the function using the three point equal interval search.
    This is assuming the function is continuous over the given interval,
    the function is unimodal, and there is a minimum to solve for. The search will iterate until
    successive approximations of x are less than 1e-5 or until 50 iterations are reached. These
     stopping criteria values can be specified by the user.
    :param steps: Number of steps until algorythm stops iterating. Defaults to 50.
    :param threshold: Minimum distance between successive estimations for x until the
     algorythm stops iterating. Defaults to 1e-5
    :param f: Function to find x coordinate of minimum value.
    :param x_min: Lower bound of interval to find minimum.
    :param x_max: Upper bound of interval to find minimum.
    :return: Approximate x coordinate where f is at a minimum.
    """
    x = np.linspace(x_min, x_max, 5)
    iteration = 0
    dist = x[4] - x[0]
    y = f(x)
    min_index = np.argmin(y)
    while iteration <= steps and threshold < dist:
        if min_index.size == 2:
            x = np.linspace(x[min_index[0]], x[min_index[1]], 5)
        elif min_index == 0:
            x = np.linspace(x[0], x[1], 5)
        elif min_index == 4:
            x = np.linspace(x[3], x[4], 5)
        else:
            x = np.linspace(x[min_index - 1], x[min_index + 1], 5)
        iteration += 1
        dist = x[4] - x[0]
        y = f(x)
        min_index = np.argmin(y)
    return float((x[0] + x[4]) / 2)


def successive_parabolic_interpolation(f: callable(float), samples: np.array,
                                       threshold: float = 1e-5, steps=30) -> float:
    """
    Given a function, and three x values to sample at, returns the approximate x value
    of the minimum value of the function using the successive parabolic interpolation search.
    This is assuming the function is continuous over the interval of the given points,
    the function is unimodal, and there is a minimum to solve for. This method
    often has issues however, and may fail. The search will iterate until
    successive approximations of x are less than 1e-5 or until 30 iterations are reached. These
    stopping criteria values can be specified by the user.
    :param steps: Number of steps until algorythm stops iterating. Defaults to 30.
    :param threshold: Minimum distance between successive estimations for x until the
     algorythm stops iterating. Defaults to 1e-5
    :param f: Function to find x coordinate of minimum value.
    :param samples: List of three x values to start sampling at.
    :return: Approximate x coordinate where f is at a minimum.
    """
    iteration = 0
    dist = samples[2] - samples[0]
    while iteration <= steps and threshold < dist:
        y = f(samples)
        a, b, c = np.polyfit(samples, y[0:3], 2)
        x_min = - b / (2 * a)
        samples = np.append(samples, x_min)
        y = f(samples)
        y_max = np.argmax(y)
        samples = np.delete(samples, y_max)
        iteration += 1
        dist = max(samples) - min(samples)
    return float((max(samples) + min(samples)) / 2)


def gradient_descent(df: callable(tuple), x_0: np.array,
                     learning_rate: float, threshold: float = 1e-5, steps=30) -> float:
    """
    Given the gradient of a function, a starting point, and a learning rate,
    returns the approximate point of the minimum value of the function using
    gradient descent search. The search will iterate until
    successive approximations of x are less than 1e-5 or until 30 iterations are reached.
    These stopping criteria values can be specified by the user.
    :param learning_rate: learning rate for the algorythm.
    :param x_0: initial starting point.
    :param steps: Number of steps until algorythm stops iterating. Defaults to 30.
    :param threshold: Minimum distance between successive estimations until the
     algorythm stops iterating. Defaults to 1e-5
    :param df: Gradient of function to minimize
    :return: Approximate input coordinates where f is at a minimum.
    """
    x_1 = x_0 - np.multiply(learning_rate, df(x_0))
    iteration = 1
    dist = np.linalg.norm(x_1 - x_0)
    while iteration < steps and threshold < dist:
        x_0 = x_1
        x_1 = x_0 - np.multiply(learning_rate, df(x_0))
        iteration += 1
        dist = np.linalg.norm(x_1 - x_0)
    return x_1
