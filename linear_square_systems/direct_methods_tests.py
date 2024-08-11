import numpy as np
import direct_methods


def test_back_substitution():
    a = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )
    b = np.array([1, 2, 3]).transpose()
    sol = np.array([1, 2, 3])
    np.testing.assert_almost_equal(direct_methods.back_substitution(a, b), sol, err_msg="Back Sub Test 1 Error")

    a = np.array(
        [[1, 1, 1],
         [0, 1, 1],
         [0, 0, 1]]
    )
    sol = np.array([-1, -1, 3])
    np.testing.assert_almost_equal(direct_methods.back_substitution(a, b), sol, err_msg="Back Sub Test 2 Error")

    a = np.array([
        [3, 2, 1],
        [0, 3, 2],
        [0, 0, 3]]
    )
    sol = np.array([0, 0, 1])
    np.testing.assert_almost_equal(direct_methods.back_substitution(a, b), sol, err_msg="Back Sub Test 3 Error")

    b = np.array([0, 0, 0]).transpose()
    sol = np.array([0, 0, 0])
    np.testing.assert_almost_equal(direct_methods.back_substitution(a, b), sol, err_msg="Back Sub Test 4 Error")

    b = np.array([1, 1, 1]).transpose()
    sol = np.array([4 / 27, 1 / 9, 1 / 3])
    np.testing.assert_almost_equal(direct_methods.back_substitution(a, b), sol, err_msg="Back Sub Test 5 Error")

    a = np.array(
        [[1, 2],
         [0, 7]]
    )
    b = np.array([2, 3]).transpose()
    sol = np.array([8 / 7, 3 / 7])
    np.testing.assert_almost_equal(direct_methods.back_substitution(a, b), sol, err_msg="Back Sub Test 6 Error")

    a = np.array(
        [[1, 2],
         [0, 0]]
    )
    b = np.array([1, 0]).transpose()
    sol = np.array([1, 0])
    np.testing.assert_almost_equal(direct_methods.back_substitution(a, b), sol, err_msg="Back Sub Test 7 Error")

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
    np.testing.assert_almost_equal(direct_methods.regular_gaussian_elim(a), sol, err_msg="Regular GE Test 1 Fail")

    a = np.array(
        [[1, 2, 3],
         [1, 3, 4],
         [1, 1, 1]]
    )
    sol = np.array(
        [[1, 2, 3],
         [0, 1, 1],
         [0, 0, -1]]
    )
    np.testing.assert_almost_equal(direct_methods.regular_gaussian_elim(a), sol, err_msg="Regular GE Test 2 Fail")

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
    np.testing.assert_almost_equal(direct_methods.regular_gaussian_elim(a), sol, err_msg="Regular GE Test 3 Fail")

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
    np.testing.assert_almost_equal(direct_methods.regular_gaussian_elim(a), sol, err_msg="Regular GE Test 4 Fail")

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
    np.testing.assert_almost_equal(direct_methods.regular_gaussian_elim(a), sol, err_msg="Regular GE Test 5 Fail")

    # Tests for error with non regular matrix
    a = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]
    )
    np.testing.assert_raises(ValueError, direct_methods.regular_gaussian_elim, a)


if __name__ == "__main__":
    test_back_substitution()
    print("Back sub passed")
    test_regular_gaussian_elimination()
    print("Regular GE passed")
    print("Tests Passed!")
