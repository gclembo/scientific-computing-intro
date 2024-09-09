import numpy as np
import initial_value_problems


def forward_approximation_tests():
    """
    Tests forward difference approximation function.
    """

    def f(x): return x ** 2

    np.testing.assert_almost_equal(
        initial_value_problems.forward_approximation(f, -1, 0.01),
        -1.99,
        err_msg="Forward Difference Approximation Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.forward_approximation(f, 0.5, 0.01),
        1.01,
        err_msg="Forward Difference Approximation Test 2 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.forward_approximation(f, 3, 0.01),
        6.01,
        err_msg="Forward Difference Approximation Test 3 Fail"
    )


def backward_approximation_tests():
    """
    Tests backward difference approximation function.
    """

    def f(x): return x ** 2

    np.testing.assert_almost_equal(
        initial_value_problems.backward_approximation(f, -1, 0.01),
        -2.01,
        err_msg="Backward Difference Approximation Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.backward_approximation(f, 0.5, 0.01),
        0.99,
        err_msg="Backward Difference Approximation Test 2 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.backward_approximation(f, 3, 0.01),
        5.99,
        err_msg="Backward Difference Approximation Test 3 Fail"
    )


def central_approximation_tests():
    """
    Tests central difference approximation function.
    """

    def f(x): return x ** 2

    np.testing.assert_almost_equal(
        initial_value_problems.central_approximation(f, -1, 0.01),
        -2,
        err_msg="Central Difference Approximation Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.central_approximation(f, 0.5, 0.01),
        1,
        err_msg="Central Difference Approximation Test 2 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.central_approximation(f, 3, 0.01),
        6,
        err_msg="Central Difference Approximation Test 3 Fail"
    )

    def f(x): return x * np.cos(x)

    np.testing.assert_almost_equal(
        initial_value_problems.central_approximation(f, -1, 0.01),
        -0.301181669,
        err_msg="Central Difference Approximation Test 4 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.central_approximation(f, 0.5, 0.01),
        0.637829909,
        err_msg="Central Difference Approximation Test 5 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.central_approximation(f, 3, 0.01),
        -1.413295965,
        err_msg="Central Difference Approximation Test 6 Fail"
    )


def forward_euler_tests():
    """
    Tests forward Euler IVP function.
    """

    def dy(t, y): return y + t

    np.testing.assert_almost_equal(
        initial_value_problems.forward_euler(dy, 1, 0, 0.1, 1),
        1.1,
        err_msg="Forward Euler Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.forward_euler(dy, 1, 0, 0.2, 2),
        1.22,
        err_msg="Forward Euler Test 2 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.forward_euler(dy, 1, 0, 0.3, 3),
        1.362,
        err_msg="Forward Euler Test 3 Fail"
    )

    np.testing.assert_almost_equal(
        initial_value_problems.forward_euler(dy, 1, 1, 1.1, 1),
        1.2,
        err_msg="Forward Euler Test 4 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.forward_euler(dy, 1, 1, 1.2, 2),
        1.43,
        err_msg="Forward Euler Test 5 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.forward_euler(dy, 1, 1, 1.3, 3),
        1.693,
        err_msg="Forward Euler Test 6 Fail"
    )


def backward_euler_tests():
    """
    Tests backward Euler IVP function.
    """

    def dy(t, y): return y + t

    np.testing.assert_almost_equal(
        initial_value_problems.backward_euler(dy, 1, 0, 0.1, 1),
        1.122222222222222,
        err_msg="Backward Euler Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.backward_euler(dy, 1, 0, 0.2, 2),
        1.269135802469136,
        err_msg="Backward Euler Test 2 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.backward_euler(dy, 1, 0, 0.3, 3),
        1.443484224965706,
        err_msg="Backward Euler Test 3 Fail"
    )

    def dy(t, y): return t * y + 1

    np.testing.assert_almost_equal(
        initial_value_problems.backward_euler(dy, 1, 0, 0.1, 1),
        1.111111111111111,
        err_msg="Backward Euler Test 4 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.backward_euler(dy, 1, 0, 0.2, 2),
        1.235827664399093,
        err_msg="Backward Euler Test 5 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.backward_euler(dy, 1, 0, 0.3, 3),
        1.377141922060921,
        err_msg="Backward Euler Test 6 Fail"
    )


def trapezoid_method_tests():
    """
    Tests trapezoid method IVP function.
    """

    def dy(t, y): return y + t

    np.testing.assert_almost_equal(
        initial_value_problems.trapezoid_method(dy, 1, 0, 0.1, 1),
        1.111111111111111,
        err_msg="Trapezoid Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.trapezoid_method(dy, 1, 0, 0.2, 2),
        1.244506172839506,
        err_msg="Trapezoid Test 2 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.trapezoid_method(dy, 1, 0, 0.3, 3),
        1.402537379972565,
        err_msg="Trapezoid Test 3 Fail"
    )

    def dy(t, y): return t * y + 1

    np.testing.assert_almost_equal(
        initial_value_problems.trapezoid_method(dy, 1, 0, 0.1, 1),
        1.105555555555556,
        err_msg="Trapezoid Test 4 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.trapezoid_method(dy, 1, 0, 0.2, 2),
        1.223384920634921,
        err_msg="Trapezoid Test 5 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.trapezoid_method(dy, 1, 0, 0.3, 3),
        1.356083485108820,
        err_msg="Trapezoid Test 6 Fail"
    )


def rk4_tests():
    """
    Tests Runge-Kutta 4 IVP function.
    """

    def dy(t, y): return y * t + 1

    np.testing.assert_almost_equal(
        initial_value_problems.rk4(dy, 1, 0, 0.1, 1),
        1.05346480208,
        err_msg="RK4 Test 1 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.rk4(dy, 1, 0, 0.2, 2),
        1.23001295039,
        err_msg="RK4 Test 2 Fail"
    )
    np.testing.assert_almost_equal(
        initial_value_problems.rk4(dy, 1, 0, 0.3, 3),
        1.62955553524,
        err_msg="RK4 Test 3 Fail"
    )


if __name__ == "__main__":
    forward_approximation_tests()
    print("Forward Difference Approximation Tests Passed")
    backward_approximation_tests()
    print("Backward Difference Approximation Tests Passed")
    central_approximation_tests()
    print("Central Difference Approximation Tests Passed")
    forward_euler_tests()
    print("Forward Euler Tests Passed")
    backward_euler_tests()
    print("Backward Euler Tests Passed")
    trapezoid_method_tests()
    print("Trapezoid Method Tests Passed")
    rk4_tests()
    print("RK4 Euler Tests Passed")
    print("Tests Passed")
