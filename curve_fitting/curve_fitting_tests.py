import numpy as np
import curve_fitting


def poly_fit_tests():
    """
    Tests polynomial least squares fitting function.
    """
    x = [0, 1, 2, 3]
    y = [0, 2, 3, 5]
    sol = np.array([0.1, 1.6])
    np.testing.assert_allclose(
        curve_fitting.poly_fit(x, y, 1),
        sol,
        err_msg="Poly Fit Test 1"
    )

    x = np.array([0, 1, 2, 3])
    y = np.array([0, 2, 3, 5])
    np.testing.assert_allclose(
        curve_fitting.poly_fit(x, y, 1),
        sol,
        err_msg="Poly Fit Test 2"
    )

    x = [0, 0.5, 3, 5.3, 8]
    y = [-1, 2, 5, 9, 20]
    sol = np.array([-0.408940808587034, 3.507190767281546, -0.760158857355591, 0.080171856705389])
    np.testing.assert_allclose(
        curve_fitting.poly_fit(x, y, 3),
        sol,
        err_msg="Poly Fit Test 3"
    )


def func_fit_tests():
    """
    Tests general least squares function fitting function.
    """
    x = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    y = [1, 1 + np.sqrt(2) / 2, 2, 1 + np.sqrt(2) / 2, 1]
    sol = np.array([1, 1])
    np.testing.assert_allclose(
        curve_fitting.func_fit(x, y, [np.sin]),
        sol,
        err_msg="Function Fitting Test 1"
    )

    x = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    y = np.array([1, 1 + np.sqrt(2) / 2, 2, 1 + np.sqrt(2) / 2, 1])
    np.testing.assert_allclose(
        curve_fitting.func_fit(x, y, [np.sin]),
        sol,
        err_msg="Function Fitting Test 2"
    )

    x = [1, 5, 9, 15, 22]
    y = [20, 12, 8, 11, 23]
    sol = np.array([20.4937691, 0.0500474, -7.2152122])
    np.testing.assert_allclose(
        curve_fitting.func_fit(x, y, [lambda a: a ** 2, np.log]),
        sol,
        rtol=1e-06,
        err_msg="Function Fitting Test 3"
    )


if __name__ == "__main__":
    poly_fit_tests()
    print("Polynomial Fitting Tests Passed")
    func_fit_tests()
    print("Function Fitting Tests Passed")
    print("Tests Passed!")
