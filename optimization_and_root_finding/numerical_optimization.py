

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


def successive_parabolic_interpolation(f: callable(float), samples: tuple,
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


def gradient_descent(df: callable(tuple), x_0: tuple,
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
