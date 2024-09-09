import numpy as np
import direct_methods


def test_back_substitution():
    """
    Tests back substitution function.
    """
    a = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )
    b = np.array([1, 2, 3]).transpose()
    sol = np.array([1, 2, 3])
    np.testing.assert_allclose(
        direct_methods.back_substitution(a, b),
        sol,
        err_msg="Back Sub Test 1 Error"
    )

    a = np.array(
        [[1, 1, 1],
         [0, 1, 1],
         [0, 0, 1]]
    )
    sol = np.array([-1, -1, 3])
    np.testing.assert_allclose(
        direct_methods.back_substitution(a, b),
        sol,
        err_msg="Back Sub Test 2 Error"
    )

    a = np.array([
        [3, 2, 1],
        [0, 3, 2],
        [0, 0, 3]]
    )
    sol = np.array([0, 0, 1])
    np.testing.assert_allclose(
        direct_methods.back_substitution(a, b),
        sol,
        err_msg="Back Sub Test 3 Error"
    )

    b = np.array([0, 0, 0]).transpose()
    sol = np.array([0, 0, 0])
    np.testing.assert_allclose(
        direct_methods.back_substitution(a, b),
        sol,
        err_msg="Back Sub Test 4 Error"
    )

    b = np.array([1, 1, 1]).transpose()
    sol = np.array([4 / 27, 1 / 9, 1 / 3])
    np.testing.assert_allclose(
        direct_methods.back_substitution(a, b),
        sol,
        err_msg="Back Sub Test 5 Error"
    )

    a = np.array(
        [[1, 2],
         [0, 7]]
    )
    b = np.array([2, 3]).transpose()
    sol = np.array([8 / 7, 3 / 7])
    np.testing.assert_allclose(
        direct_methods.back_substitution(a, b),
        sol,
        err_msg="Back Sub Test 6 Error"
    )

    a = np.array(
        [[1, 2],
         [0, 0]]
    )
    b = np.array([1, 0]).transpose()
    sol = np.array([1, 0]).transpose()
    np.testing.assert_allclose(
        direct_methods.back_substitution(a, b),
        sol,
        err_msg="Back Sub Test 7 Error"
    )

    # check error if not upper triangular
    a = np.array(
        [[1, 2],
         [1, 7]]
    )

    np.testing.assert_raises(ValueError, direct_methods.back_substitution, a, b)

    # Checks if matrix is not square
    a = np.array(
        [[1, 2, 3],
         [0, 7, 4]]
    )
    np.testing.assert_raises(ValueError, direct_methods.back_substitution, a, b)

    # checks error if a and b are not the same height
    a = np.array(
        [[1, 2, 5],
         [0, 7, 4],
         [1, 2, 3]]
    )
    np.testing.assert_raises(ValueError, direct_methods.back_substitution, a, b)

    # checks error if no sols
    a = np.array(
        [[1, 2],
         [0, 0]]
    )
    b = np.array([1, 1]).transpose()
    np.testing.assert_raises(ValueError, direct_methods.back_substitution, a, b)


def test_regular_gaussian_elimination():
    """
    Tests regular Gaussian elimination function.
    """
    a = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )
    sol = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )
    np.testing.assert_allclose(
        direct_methods.regular_gaussian_elim(a),
        sol,
        err_msg="Regular GE Test 1 Fail"
    )

    a = np.array(
        [[2, 2, 3],
         [1, 3, 4],
         [1, 1, 1]]
    )
    sol = np.array(
        [[2, 2, 3],
         [0, 2, 2.5],
         [0, 0, -0.5]]
    )
    np.testing.assert_allclose(
        direct_methods.regular_gaussian_elim(a),
        sol,
        err_msg="Regular GE Test 2 Fail"
    )

    a = np.array(
        [[1, 2, 3],
         [1, 1, 1],
         [1, 1, 1]]
    )
    sol = np.array(
        [[1, 2, 3],
         [0, -1, -2],
         [0, 0, 0]]
    )
    np.testing.assert_allclose(
        direct_methods.regular_gaussian_elim(a),
        sol,
        err_msg="Regular GE Test 3 Fail"
    )

    a = np.array(
        [[1, 2],
         [1, 3],
         [1, 1]]
    )
    sol = np.array(
        [[1, 2],
         [0, 1],
         [0, 0]]
    )
    np.testing.assert_allclose(
        direct_methods.regular_gaussian_elim(a),
        sol,
        err_msg="Regular GE Test 4 Fail"
    )

    a = np.array(
        [[1, 2, 3, 4],
         [1, 3, 4, 5],
         [1, 1, 1, 1]]
    )
    sol = np.array(
        [[1, 2, 3, 4],
         [0, 1, 1, 1],
         [0, 0, -1, -2]]
    )
    np.testing.assert_allclose(
        direct_methods.regular_gaussian_elim(a),
        sol,
        err_msg="Regular GE Test 5 Fail"
    )

    # Tests for error with non regular matrix
    a = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]
    )
    np.testing.assert_raises(ValueError, direct_methods.regular_gaussian_elim, a)


def test_complete_gaussian_elimination():
    """
    Tests complete Gaussian elimination function.
    """
    a = np.array(
        [[2, 2, 3],
         [1, 1, 1],
         [1, 1, 1]]
    )
    sol = np.array(
        [[2, 2, 3],
         [0, 0, -0.5],
         [0, 0, 0]]
    )
    np.testing.assert_allclose(
        direct_methods.complete_gaussian_elim(a),
        sol,
        err_msg="Complete GE Test 1 Fail"
    )

    a = np.array(
        [[1, 2],
         [1, 3],
         [1, 1]]
    )
    sol = np.array(
        [[1, 2],
         [0, 1],
         [0, 0]]
    )
    np.testing.assert_allclose(
        direct_methods.complete_gaussian_elim(a),
        sol,
        err_msg="Complete GE Test 2 Fail"
    )

    a = np.array(
        [[1, 2, 3, 4],
         [1, 3, 4, 5],
         [1, 1, 1, 1]]
    )
    sol = np.array(
        [[1, 2, 3, 4],
         [0, 1, 1, 1],
         [0, 0, -1, -2]]
    )
    np.testing.assert_allclose(
        direct_methods.complete_gaussian_elim(a),
        sol,
        err_msg="Complete GE Test 3 Fail"
    )

    a = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 2, 3]]
    )
    sol = np.array(
        [[1, 1, 1],
         [0, 1, 2],
         [0, 0, 0]]
    )
    np.testing.assert_allclose(
        direct_methods.complete_gaussian_elim(a),
        sol,
        err_msg="Complete GE Test 4 Fail"
    )

    a = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]
    )
    sol = np.array(
        [[1, 1, 1],
         [0, 0, 0],
         [0, 0, 0]]
    )
    np.testing.assert_allclose(
        direct_methods.complete_gaussian_elim(a),
        sol,
        err_msg="Complete GE Test 5 Fail"
    )

    a = np.array(
        [[1, 1, 2, 1, 1],
         [1, 1, 1, 3, 2],
         [1, 1, 2, 1, 3],
         [1, 1, 1, 1, 4]]
    )
    sol = np.array(
        [[1, 1, 2, 1, 1],
         [0, 0, -1, 0, 3],
         [0, 0, 0, 2, -2],
         [0, 0, 0, 0, 2]]
    )
    np.testing.assert_allclose(
        direct_methods.complete_gaussian_elim(a),
        sol,
        err_msg="Complete GE Test 6 Fail"
    )


def test_lu_decomposition():
    """
    Tests LU decomposition function.
    """
    a = np.array(
        [[2, 1],
         [7, 3]]
    )
    sol_l = np.array(
        [[1, 0],
         [3.5, 1]]
    )
    sol_u = np.array(
        [[2, 1],
         [0, -0.5]]
    )
    l, u = direct_methods.lu_decomposition(a)
    np.testing.assert_allclose(
        l,
        sol_l,
        err_msg="LU decomposition L test 1 Fail"
    )
    np.testing.assert_allclose(
        u,
        sol_u,
        err_msg="LU decomposition U test 1 Fail"
    )

    a = np.array(
        [[5, 3, 6],
         [5, 4, 2],
         [2, 6, 4]]
    )
    sol_l = np.array(
        [[1, 0, 0],
         [1, 1, 0],
         [2 / 5, 24 / 5, 1]]
    )
    sol_u = np.array(
        [[5, 3, 6],
         [0, 1, -4],
         [0, 0, 104 / 5]]
    )
    l, u = direct_methods.lu_decomposition(a)
    np.testing.assert_allclose(
        l,
        sol_l,
        err_msg="LU decomposition L test 2 Fail"
    )
    np.testing.assert_allclose(
        u,
        sol_u,
        err_msg="LU decomposition U test 2 Fail"
    )

    # Test Error with non-regular matrix
    a = np.array(
        [[5, 3, 6],
         [5, 3, 2],
         [2, 6, 4]]
    )

    np.testing.assert_raises(ValueError, direct_methods.lu_decomposition, a)

    # Test Error with non-square matrix
    a = np.array(
        [[1, 2, 5],
         [3, 4, 3]]
    )

    np.testing.assert_raises(ValueError, direct_methods.lu_decomposition, a)


if __name__ == "__main__":
    test_back_substitution()
    print("Back sub tests passed")
    test_regular_gaussian_elimination()
    print("Regular GE tests passed")
    test_complete_gaussian_elimination()
    print("Complete GE tests passed")
    test_lu_decomposition()
    print("LU decomposition tests passed")
    print("Tests Passed!")
