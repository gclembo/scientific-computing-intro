import numerical_optimization
import numpy.testing


def three_point_search_tests():
    def f(x): (x - 1) ** 2

    x_min = -8
    x_max = 8
    numpy.testing.assert_almost_equal(
        numerical_optimization.three_point_search(f, x_min, x_max, steps=1),
        0,
        err_msg="Three Point Interval Search Test 1 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.three_point_search(f, x_min, x_max, steps=2),
        1,
        err_msg="Three Point Interval Search Test 2 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.three_point_search(f, x_min, x_max, steps=3),
        1,
        err_msg="Three Point Interval Search Test 3 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.three_point_search(f, x_min, x_max, threshold=1),
        1,
        err_msg="Three Point Interval Search Test 4 Fail"
    )

    def f(x): 1 - x + x ** 3
    x_min = 0
    x_max = 2
    numpy.testing.assert_almost_equal(
        numerical_optimization.three_point_search(f, x_min, x_max, threshold=0.001),
        0.5772,
        err_msg="Three Point Interval Search Test 5 Fail"
    )


def successive_parabolic_interpolation_tests():
    def f(x): 1 - x + x ** 3
    samples = (0.5, 1, 1.5)
    numpy.testing.assert_almost_equal(
        numerical_optimization.successive_parabolic_interpolation(f, samples, threshold=0.001),
        0.57735,
        err_msg="SPI Test 1 Fail"
    )


def gradient_descent_tests():
    def df(x): 2 * x[0] - 2

    x_0 = (0,)
    gamma = 0.25
    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, steps=1),
        0.5,
        err_msg="Gradient Descent Test 1 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, steps=2),
        0.75,
        err_msg="Gradient Descent Test 2 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, steps=3),
        0.875,
        err_msg="Gradient Descent Test 3 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, steps=4),
        0.9375,
        err_msg="Gradient Descent Test 4 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, threshold=0.1),
        0.9375,
        err_msg="Gradient Descent Test 5 Fail"
    )

    def df(x): (2 * x[0], 2 * (x[1] - 1))
    x_0 = (0, 0)
    gamma = 0.25

    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, steps=1),
        (0, 0.5),
        err_msg="Gradient Descent Test 6 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, steps=2),
        (0, 0.75)
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, steps=3),
        (0, 0.875),
        err_msg="Gradient Descent Test 7 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, steps=4),
        (0, 0.9375),
        err_msg="Gradient Descent Test 8 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, threshold=0.1),
        (0, 0.9375),
        err_msg="Gradient Descent Test 9 Fail"
    )
    numpy.testing.assert_almost_equal(
        numerical_optimization.gradient_descent(df, x_0, gamma, threshold=1e-10),
        (0, 0.999999999941792),
        err_msg="Gradient Descent Test 10 Fail"
    )


if __name__ == '__main__':
    three_point_search_tests()
    print("Three Point Search Passed")
    successive_parabolic_interpolation_tests()
    print("SPI Passed")
    gradient_descent_tests()
    print("Gradient Descent Passed")
    print("Tests Passed")
