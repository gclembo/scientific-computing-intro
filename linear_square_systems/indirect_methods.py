import numpy as np


def neumann_iteration(input_matrix: np.array, b: np.array, x0: np.array = None, threshold: float = 10 ** (-5),
                      steps=30) -> np.array:
    """
    This function performs Nuemann Iteration given a system and an output vector b. The iterations
    stops after 30 steps or when the cauchy error is less than 10^-5 by default but these values
    can be specified. The final vector is returned
    :param input_matrix:
    :param b:
    :param x0:
    :param threshold:
    :param steps:
    :return:
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    iter_mat = np.eye(input_matrix.shape[0]) - input_matrix
    iteration = 1
    x1 = np.matmul(iter_mat, x0) + b
    dist = np.linalg.norm(x1 - x0)

    while iteration <= steps and dist >= threshold:
        x0 = x1
        x1 = np.matmul(iter_mat, x0) + b
        dist = np.linalg.norm(x1 - x0)
        iteration += 1

    if iteration <= steps:
        print("Iteration stopped after succesive solutoins were closer than " + str(threshold))
    else:
        print("Iteration stopped after " + str(steps) + " steps")

    predicted = np.matmul(input_matrix, x1)
    output_dist = np.linalg.norm(predicted - b)
    print("The vector arrived at: " + str(x1) + " predicts a b vector of " + str(predicted))
    print("The distance between actual: " + str(b) + " and predicted is " + str(output_dist))
    return x1

def jacobi_iteration(input_matrix: np.array, b: np.array, x0: np.array = None, threshold: float = 10 ** (-5),
                      steps=30) -> np.array:
    """
    :param input_matrix:
    :param b:
    :param x0:
    :param threshold:
    :param steps:
    :return:
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    lower = np.tril(input_matrix, -1)
    upper = np.triu(input_matrix, 1)
    diagonal = np.diagflat(np.diagonal(input_matrix))
    diag_inverse = np.linalg.inv(diagonal)
    iter_mat = -np.matmul(diag_inverse, (lower + upper))
    c = np.matmul(diag_inverse, b)

    iteration = 1
    x1 = np.matmul(iter_mat, x0) + c
    dist = np.linalg.norm(x1 - x0)
    print(x1)

    while iteration <= steps and dist >= threshold:
        x0 = x1
        x1 = np.matmul(iter_mat, x0) + c
        dist = np.linalg.norm(x1 - x0)
        iteration += 1
        print(x1)

    if iteration <= steps:
        print("Iteration stopped after succesive solutoins were closer than " + str(threshold))
    else:
        print("Iteration stopped after " + str(steps) + " steps")

    predicted = np.matmul(input_matrix, x1)
    output_dist = np.linalg.norm(predicted - b)
    print("The vector arrived at: " + str(x1) + " predicts a b vector of " + str(predicted))
    print("The distance between actual: " + str(b) + " and predicted is " + str(output_dist))
    return x1


a = np.array([[2, 1], [1, 2]])
b = np.array([6, 6]).transpose()
print(jacobi_iteration(a, b))
