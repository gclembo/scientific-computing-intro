import numpy as np
import direct_methods

def test_back_substitution():
    a1 = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )
    b = np.array([1, 2, 3]).transpose()
    sol = np.array([1, 2, 3])
    assert np.array_equal(direct_methods.back_substitution(a1, b), sol)

    a2 = np.array(
        [[1, 1, 1],
         [0, 1, 1],
         [0, 0, 1]]
    )
    sol = np.array([-1, -1, 3])
    assert np.array_equal(direct_methods.back_substitution(a2, b), sol)

    a3 = np.array([
        [3, 2, 1],
        [0, 3, 2],
        [0, 0, 3]]
    )
    sol = np.array([0, 0, 1])
    assert np.array_equal(direct_methods.back_substitution(a3, b), sol)


if __name__ == "__main__":
    test_back_substitution()
    print("Tests Passed!")