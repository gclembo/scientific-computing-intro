import numpy as np
import root_finding


def bisection_method_tests():
    def f(x): return x - x ** 3 + 1

    np.testing.assert_almost_equal(
        root_finding.bisection_method(f, 0, 2, steps=1),
        3 / 2
    )
    np.testing.assert_almost_equal(
        root_finding.bisection_method(f, 0, 2, steps=2),
        5 / 4
    )
    np.testing.assert_almost_equal(
        root_finding.bisection_method(f, 0, 2, steps=3),
        11 / 8
    )
    np.testing.assert_almost_equal(
        root_finding.bisection_method(f, 0, 2),
        1.2471795724
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
    def f(x): return x - x ** 3 + 1

    def df(x): return 1 - 3 * x ** 2

    np.testing.assert_almost_equal(
        root_finding.newton_method(f, df, 1, steps=1),
        3 / 2
    )
    np.testing.assert_almost_equal(
        root_finding.newton_method(f, df, 1, steps=2),
        31 / 23
    )
    np.testing.assert_almost_equal(
        root_finding.newton_method(f, df, 1, steps=3),
        71749 / 54142
    )
    np.testing.assert_almost_equal(
        root_finding.bisection_method(f, 0, 2),
        1.2471795724
    )
    np.testing.assert_raises(
        ValueError,
        root_finding.newton_method,
        f, df, np.sqrt(1/3), 0.1, 5
    )


def secant_method_tests():
    def f(x): return x - x ** 3 + 1

    np.testing.assert_almost_equal(
        root_finding.secant_method(f, 1, 3 / 2, steps=1),
        19 / 15
    )
    np.testing.assert_almost_equal(
        root_finding.secant_method(f, 1, 3 / 2, steps=2),
        5631 / 4279
    )
    np.testing.assert_almost_equal(
        root_finding.secant_method(f, 1, 3 / 2, steps=3),
        1.3252
    )
    np.testing.assert_almost_equal(
        root_finding.secant_method(f, 1, 3 / 2),
        1.2471795724
    )
    np.testing.assert_raises(
        ValueError,
        root_finding.secant_method,
        f, -1, 0, 0.1, 5
    )
