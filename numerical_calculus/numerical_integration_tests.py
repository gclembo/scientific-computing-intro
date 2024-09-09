import numpy as np
import numerical_integration


def left_hand_sum_tests():
    """
    Tests left hand Riemann sum function.
    """

    def f(x): return x ** 2

    np.testing.assert_almost_equal(
        numerical_integration.left_hand_sum(f, -1, 1, 4),
        3 / 4,
        err_msg="Left Hand Sum Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        numerical_integration.left_hand_sum(f, -1, 1, 100),
        0.6668,
        err_msg="Left Hand Sum Test 2 Fail"
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, 1, -1, 5
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, -1, 1, -5
    )


def right_hand_sum_tests():
    """
    Tests right hand Riemann sum function.
    """

    def f(x): return x ** 2

    np.testing.assert_almost_equal(
        numerical_integration.right_hand_sum(f, -1, 1, 4),
        3 / 4,
        err_msg="Right Hand Sum Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        numerical_integration.right_hand_sum(f, -1, 1, 100),
        0.6668,
        err_msg="Right Hand Sum Test 2 Fail"
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, 1, -1, 5
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, -1, 1, -5
    )


def midpoint_sum_tests():
    """
    Tests midpoint Riemann sum function.
    """

    def f(x): return x ** 2

    np.testing.assert_almost_equal(
        numerical_integration.midpoint_sum(f, -1, 1, 4),
        5 / 8,
        err_msg="Midpoint Sum Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        numerical_integration.midpoint_sum(f, -1, 1, 100),
        0.6666,
        err_msg="Midpoint Sum Test 2 Fail"
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, 1, -1, 5
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, -1, 1, -5
    )


def trapezoid_sum_tests():
    """
    Tests trapezoid sum function.
    """

    def f(x): return x ** 2

    np.testing.assert_almost_equal(
        numerical_integration.trapezoid_sum(f, -1, 1, 4),
        3 / 4,
        err_msg="Trapezoid Sum Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        numerical_integration.trapezoid_sum(f, -1, 1, 100),
        0.6668,
        err_msg="Trapezoid Sum Test 2 Fail"
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, 1, -1, 5
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, -1, 1, -5
    )


def simpson_rule_sum_tests():
    """
    Tests Simpson's rule sum function
    """

    def f(x): return x ** 2

    np.testing.assert_almost_equal(
        numerical_integration.simpson_rule_sum(f, -1, 1, 4),
        2 / 3,
        err_msg="Simpson\'s Rule Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        numerical_integration.simpson_rule_sum(f, -1, 1, 100),
        2 / 3,
        err_msg="Simpson\'s Rule Test 2 Fail"
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, 1, -1, 5
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, -1, 1, -5
    )
    np.testing.assert_raises(
        ValueError,
        numerical_integration.simpson_rule_sum,
        f, -1, 1, 5
    )


if __name__ == "__main__":
    left_hand_sum_tests()
    print("Left Hand Tests Passed")
    right_hand_sum_tests()
    print("Right Hand Tests Passed")
    midpoint_sum_tests()
    print("Midpoint Tests Passed")
    trapezoid_sum_tests()
    print("Trapezoid Tests Passed")
    simpson_rule_sum_tests()
    print("Simpson\'s Tests Passed")
    print("Tests Passed")
