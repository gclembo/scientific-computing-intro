import numpy as np


def three_point_search(f: callable(float), a: float, b: float,
                       threshold: float = 1e-5, iterations: int = 50) -> float:
    """
    Given a function, a minimum x value, and a maximum x value, returns the approximate x value
    of the minimum value of the function using the three point equal interval search.
    This is assuming the function is continuous over the given interval,
    the function is unimodal, and there is a minimum to solve for. The search will iterate until
    successive approximations of x are less than 1e-5 or until 50 iterations are reached. These
    stopping criteria values can be specified by the user.

    Parameters
    ----------
    f : callable(float)
        Function to find x coordinate of minimum value.
    a : float
        Lower x bound of interval to find minimum.
    b : float
        Upper x bound of interval to find minimum.
    threshold : float, default 1e-5
        Minimum distance between successive estimations for x until the
        algorythm stops iterating. Defaults to 1e-5.
    iterations : int, default 50
         Number of iterations until algorythm stops iterating. Defaults to 50.

    Returns
    -------
    float
        Approximate x coordinate where f is at a minimum.
    """
    x = np.linspace(a, b, 5)
    iteration = 0
    dist = x[4] - x[0]
    y = f(x)
    min_index = np.argmin(y)
    while iteration <= iterations and threshold < dist:
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
                                       threshold: float = 1e-5, iterations: int = 30) -> float:
    """
    Given a function, and three x values to sample at, returns the approximate x value
    of the minimum value of the function using the successive parabolic interpolation search.
    This is assuming the function is continuous over the interval of the given points,
    the function is unimodal, and there is a minimum to solve for. This method
    often has issues however, and may fail. The search will iterate until
    successive approximations of x are less than 1e-5 or until 30 iterations are reached. These
    stopping criteria values can be specified by the user.

    Parameters
    ----------
    f : callable(float)
        Function to find x coordinate of minimum value.
    samples : np.array
        Vector of three x values to start sampling at.
    threshold : float, default 1e-5
        Minimum distance between successive estimations for x until the algorythm stops iterating.
        Defaults to 1e-5
    iterations : int, default 30
        Number of iterations until algorythm stops iterating. Defaults to 30.

    Returns
    -------
    float
        Approximate x coordinate where f is at a minimum.
    """
    iteration = 0
    dist = samples[2] - samples[0]
    while iteration <= iterations and threshold < dist:
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


def gradient_descent(df: callable(np.array), x_0: np.array,
                     learning_rate: float, threshold: float = 1e-5, iterations: int = 30) -> float:
    """
    Given the gradient of a function, a starting point, and a learning rate,
    returns the approximate point of the minimum value of the function using
    gradient descent search. The search will iterate until
    successive approximations of x are less than 1e-5 or until 30 iterations are reached.
    These stopping criteria values can be specified by the user.

    Parameters
    ----------
    df : callable(np.array)
        Gradient of function to minimize
    x_0 : np.array
        Initial starting input.
    learning_rate : float
        Learning rate for the algorythm.
    threshold : float, default 1e-5
        Minimum distance between successive estimations for x until the algorythm stops iterating.
        Defaults to 1e-5
    iterations : int, default 30
        Number of iterations until algorythm stops iterating. Defaults to 30.

    Returns
    -------
    float
        Approximate input coordinates where f is at a minimum.
    """
    x_1 = x_0 - np.multiply(learning_rate, df(x_0))
    iteration = 1
    dist = np.linalg.norm(x_1 - x_0)
    while iteration < iterations and threshold < dist:
        x_0 = x_1
        x_1 = x_0 - np.multiply(learning_rate, df(x_0))
        iteration += 1
        dist = np.linalg.norm(x_1 - x_0)
    return x_1
