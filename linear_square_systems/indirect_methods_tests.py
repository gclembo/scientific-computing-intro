import numpy as np
import indirect_methods


def is_diagonally_dominant_tests():
    a = np.array([[2, 1], [1, 2]])
    np.testing.assert_equal(indirect_methods.is_diagonally_dominant(a), True,
                            err_msg="Diagonally Dominant Test 1 Fail")

    a = np.array([[2, 1, 0], [0, 2, 1], [0, 1, 2]])
    np.testing.assert_equal(indirect_methods.is_diagonally_dominant(a), True,
                            err_msg="Diagonally Dominant Test 2 Fail")

    a = np.array([[2, -1, 0], [0, -2, -1], [0, 1, -2]])
    np.testing.assert_equal(indirect_methods.is_diagonally_dominant(a), True,
                            err_msg="Diagonally Dominant Test 3 Fail")

    a = np.array([[1, 2], [1, 2]])
    np.testing.assert_equal(indirect_methods.is_diagonally_dominant(a), False,
                            err_msg="Diagonally Dominant Test 4 Fail")

    a = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    np.testing.assert_equal(indirect_methods.is_diagonally_dominant(a), False,
                            err_msg="Diagonally Dominant Test 5 Fail")

    a = np.array([[2, -1, 1], [-1, 2, -1], [1, 1, -2]])
    np.testing.assert_equal(indirect_methods.is_diagonally_dominant(a), False,
                            err_msg="Diagonally Dominant Test 6 Fail")


def neumann_iteration_tests():
    a = np.array([[1, 0.5], [0.5, 1]])
    b = np.array([3, 3]).transpose()
    np.testing.assert_allclose(indirect_methods.neumann_iteration(a, b, steps=1),
                               np.array([3, 3]).transpose(), err_msg="Neumann Test 1a")
    np.testing.assert_allclose(indirect_methods.neumann_iteration(a, b, steps=3),
                               np.array([2.25, 2.25]).transpose(), err_msg="Neumann Test 1b")
    np.testing.assert_allclose(indirect_methods.neumann_iteration(a, b, steps=5),
                               np.array([2.0625, 2.0625]).transpose(), err_msg="Neumann Test 1c")

    a = np.array([[2, 1], [1, 2]])
    b = np.array([6, 6]).transpose()
    np.testing.assert_allclose(indirect_methods.neumann_iteration(a, b, steps=1),
                               np.array([6, 6]).transpose(), err_msg="Neumann Test 2a")
    np.testing.assert_allclose(indirect_methods.neumann_iteration(a, b, steps=3),
                               np.array([18, 18]).transpose(), err_msg="Neumann Test 2b")
    np.testing.assert_allclose(indirect_methods.neumann_iteration(a, b, steps=5),
                               np.array([66, 66]).transpose(), err_msg="Neumann Test 2c")

    a = np.array([[2, 1], [1, 2]])
    b = np.array([6, 6]).transpose()
    x0 = np.array([0, 2]).transpose()
    np.testing.assert_allclose(indirect_methods.neumann_iteration(a, b, steps=1, x0=x0),
                               np.array([4, 4]).transpose(), err_msg="Neumann Test 3a")
    np.testing.assert_allclose(indirect_methods.neumann_iteration(a, b, steps=3, x0=x0),
                               np.array([10, 10]).transpose(), err_msg="Neumann Test 3b")
    np.testing.assert_allclose(indirect_methods.neumann_iteration(a, b, steps=5, x0=x0),
                               np.array([34, 34]).transpose(), err_msg="Neumann Test 3c")


def jacobi_iteration_tests():
    a = np.array([[2, 1], [1, 2]])
    b = np.array([6, 6]).transpose()
    np.testing.assert_allclose(indirect_methods.jacobi_iteration(a, b, steps=1),
                               np.array([3, 3]).transpose(), err_msg="Jacobi Test 1a")
    np.testing.assert_allclose(indirect_methods.jacobi_iteration(a, b, steps=3),
                               np.array([2.25, 2.25]).transpose(), err_msg="Jacobi Test 1b")
    np.testing.assert_allclose(indirect_methods.jacobi_iteration(a, b, steps=5),
                               np.array([2.0625, 2.0625]).transpose(), err_msg="Jacobi Test 1c")


def gauss_seidel_iteration_tests():
    a = np.array([[2, 1], [1, 2]])
    b = np.array([6, 6]).transpose()
    np.testing.assert_allclose(indirect_methods.gauss_seidel_iteration(a, b, steps=1),
                               np.array([3, 3 / 2]).transpose(), err_msg="Gauss Seidel Test 1a")
    np.testing.assert_allclose(indirect_methods.gauss_seidel_iteration(a, b, steps=3),
                               np.array([66 / 32, 63 / 32]).transpose(), err_msg="Gauss Seidel Test 1b")
    np.testing.assert_allclose(indirect_methods.gauss_seidel_iteration(a, b, steps=5),
                               np.array([1026 / 512, 1023 / 512]).transpose(), err_msg="Gauss Seidel Test 1c")


def sor_iteration_tests():
    a = np.array([[2, 1], [1, 2]])
    b = np.array([6, 6]).transpose()
    np.testing.assert_allclose(indirect_methods.sor_iteration(a, b, steps=5, w=1.07),
                               np.array([2.0002185, 1.9999319]).transpose(), err_msg="SOR Test 1")


if __name__ == '__main__':
    is_diagonally_dominant_tests()
    print("Diagonally Dominant Tests Passed")
    neumann_iteration_tests()
    print("Neumann Iteration Tests Passed")
    jacobi_iteration_tests()
    print("Jacobi Iteration Tests Passed")
    gauss_seidel_iteration_tests()
    print("Gauss Seidel Iteration Tests Passed")
    sor_iteration_tests()
    print("SOR Iteration Tests Passed")
    print("Tests passed")
