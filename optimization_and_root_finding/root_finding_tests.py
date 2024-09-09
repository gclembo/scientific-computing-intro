import numpy as np
import root_finding


def bisection_method_tests():
    """
    Tests bisection method function.
    """

    def f(x): return x - x ** 3 + 1

    np.testing.assert_almost_equal(
        root_finding.bisection_method(f, 0, 2, iterations=1),
        3 / 2,
        err_msg="Bisection Method Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        root_finding.bisection_method(f, 0, 2, iterations=2),
        5 / 4,
        err_msg="Bisection Method Test 2 Fail"
    )
    np.testing.assert_almost_equal(
        root_finding.bisection_method(f, 0, 2, iterations=3),
        11 / 8,
        err_msg="Bisection Method Test 3 Fail"
    )
    np.testing.assert_almost_equal(
        root_finding.bisection_method(f, 0, 2, threshold=1e-7),
        1.324717957244746,
        err_msg="Bisection Method Test 4 Fail"
    )
    np.testing.assert_raises(
        ValueError,
        root_finding.bisection_method,
        f, -1, 0, 0.1, 5
    )
    np.testing.assert_raises(
        ValueError,
        root_finding.bisection_method,
        f, 1, -1, 0.1, 5
    )


def newton_method_tests():
    """
    Tests Newton's method function.
    """

    def f(x): return x - x ** 3 + 1

    def df(x): return 1 - 3 * x ** 2

    np.testing.assert_almost_equal(
        root_finding.newton_method(f, df, 1, iterations=1),
        3 / 2,
        err_msg="Newton\'s Method Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        root_finding.newton_method(f, df, 1, iterations=2),
        31 / 23,
        err_msg="Newton\'s Method Test 2 Fail"
    )
    np.testing.assert_almost_equal(
        root_finding.newton_method(f, df, 1, iterations=3),
        71749 / 54142,
        err_msg="Newton\'s Method Test 3 Fail"
    )
    np.testing.assert_almost_equal(
        root_finding.bisection_method(f, 0, 2, threshold=1e-7),
        1.324717957244746,
        err_msg="Newton\'s Method Test 4 Fail"
    )
    np.testing.assert_raises(
        ValueError,
        root_finding.newton_method,
        f, df, np.sqrt(1 / 3), 0.1, 5
    )


def secant_method_tests():
    """
    Tests secant method function.
    """

    def f(x): return x - x ** 3 + 1

    np.testing.assert_almost_equal(
        root_finding.secant_method(f, 1, 3 / 2, iterations=1),
        19 / 15,
        err_msg="Secant Method Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        root_finding.secant_method(f, 1, 3 / 2, iterations=2),
        5631 / 4279,
        err_msg="Secant Method Test 2 Fail"
    )
    np.testing.assert_almost_equal(
        root_finding.secant_method(f, 1, 3 / 2, iterations=3),
        1.32521411396,
        err_msg="Secant Method Test 3 Fail"
    )
    np.testing.assert_almost_equal(
        root_finding.secant_method(f, 1, 3 / 2),
        1.324717957244746,
        err_msg="Secant Method Test 4 Fail"
    )
    np.testing.assert_raises(
        ValueError,
        root_finding.secant_method,
        f, -1, 0, 0.1, 5
    )


if __name__ == "__main__":
    bisection_method_tests()
    print("Bisection Method Tests Passed")
    newton_method_tests()
    print("Newton\'s Method Tests Passed")
    secant_method_tests()
    print("Secant Method Tests Passed")
    print("Tests Passed")
