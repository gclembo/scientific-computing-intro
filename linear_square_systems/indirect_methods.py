import numpy as np


def iterate_matrix(input_matrix: np.array, b: np.array, x0: np.array = None, threshold: float = 10 ** (-5),
                   steps=30) -> np.array:
    """
    Given a matrix to multiply current approximation by in iteration scheme, a b vector to add in
    iteration scheme, an initial guess, given minimum threshold for consecutive approximations
    to stop iterating, and a maximum number of iterations to stop
    iterating, multiplies initial guess by iteration matrix and adds b vector each iteration
    until a stop condition is reached. Minimum threshold, maximum number of iterations, and initial
    guess can be specified but default to 10^-5, 30, and the zero vector respectively.
    :param input_matrix: Matrix to multiply current approximation by in iteration.
    :param b: Vector to add in iteration scheme.
    :param x0: Initial guess for iteration.
    :param threshold: Minimum distance between two vector approximations for iteration to stop.
    :param steps: Maximum number of iterations before stopping.
    :return: Vector of final iteration output.
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    iteration = 1
    x1 = np.matmul(input_matrix, x0) + b
    dist = np.linalg.norm(x1 - x0)

    while iteration < steps and dist >= threshold:
        x0 = x1
        x1 = np.matmul(input_matrix, x0) + b
        dist = np.linalg.norm(x1 - x0)
        iteration += 1

    if iteration <= steps:
        print("Iteration stopped after successive solutions were closer than " + str(threshold))
        print("This took " + str(iteration) + " steps")
    else:
        print("Iteration stopped after " + str(steps) + " steps")

    return x1


def is_diagonally_dominent(input_matrix: np.array) -> bool:
    for i in range(input_matrix.shape[0]):
        if 2 * abs(input_matrix[i][i]) <= sum(abs(input_matrix[i])):
            return False
    return True


def conclude(input_matrix: np.array, b: np.array, final: np.array) -> None:
    """
    Given a matrix representing the system that was attempted to be solved, a b vector representing
    the output vector of the system, and a final approximation of the solution to the system,
    prints conclusion for the user to interpret.
    :param input_matrix: Matrix representing system that was attempted to be solved.
    :param b: Vector of outputs of system.
    :param final: Final approximation of system solution.
    """
    predicted = np.matmul(input_matrix, final)
    output_dist = np.linalg.norm(predicted - b)
    print("The vector arrived at: " + str(final) + " predicts a b vector of " + str(predicted))
    print("The distance between actual: " + str(b) + " and predicted is " + str(output_dist))


def neumann_iteration(input_matrix: np.array, b: np.array, x0: np.array = None, threshold: float = 10 ** (-5),
                      steps=30) -> np.array:
    """
    Given a matrix representing the system to be solved, a b vector representing
    the output vector of the system, an initial guess, given minimum threshold
    for consecutive approximations
    to stop iterating, and a maximum number of iterations to stop
    iterating, performs Neumann iteration scheme on system to approximate a solution
    until a stop condition is reached. Minimum threshold, maximum number of iterations, and initial
    guess can be specified but default to 10^-5, 30, and the zero vector respectively. The final
    approximation reached is returned.
    :param input_matrix: Matrix representing system to solve.
    :param b: Vector of system outputs.
    :param x0: Initial guess for solution.
    :param threshold: Minimum distance between two vector approximations for iteration to stop.
    :param steps: Maximum number of iterations before stopping.
    :return: Vector of final iteration output.
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    iter_mat = np.eye(input_matrix.shape[0]) - input_matrix
    final = iterate_matrix(iter_mat, b, x0=x0, threshold=threshold, steps=steps)
    conclude(input_matrix, b, final)
    return final


def jacobi_iteration(input_matrix: np.array, b: np.array, x0: np.array = None, threshold: float = 10 ** (-5),
                     steps=30) -> np.array:
    """
    Given a matrix representing the system to be solved, a b vector representing
    the output vector of the system, an initial guess, given minimum threshold
    for consecutive approximations
    to stop iterating, and a maximum number of iterations to stop
    iterating, performs Jacobi iteration scheme on system to approximate a solution
    until a stop condition is reached. Minimum threshold, maximum number of iterations, and initial
    guess can be specified but default to 10^-5, 30, and the zero vector respectively. The final
    approximation reached is returned.
    :param input_matrix: Matrix representing system to solve.
    :param b: Vector of system outputs.
    :param x0: Initial guess for solution.
    :param threshold: Minimum distance between two vector approximations for iteration to stop.
    :param steps: Maximum number of iterations before stopping.
    :return: Vector of final iteration output.
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    lower = np.tril(input_matrix, -1)
    upper = np.triu(input_matrix, 1)
    diagonal = np.diagflat(np.diagonal(input_matrix))
    diag_inverse = np.linalg.inv(diagonal)
    iter_mat = -np.matmul(diag_inverse, (lower + upper))
    c = np.matmul(diag_inverse, b)

    final = iterate_matrix(iter_mat, c, x0=x0, threshold=threshold, steps=steps)
    conclude(input_matrix, b, final)
    return final


def gauss_seidel_iteration(input_matrix: np.array, b: np.array, x0: np.array = None, threshold: float = 10 ** (-5),
                           steps=30) -> np.array:
    """
    Given a matrix representing the system to be solved, a b vector representing
    the output vector of the system, an initial guess, given minimum threshold
    for consecutive approximations
    to stop iterating, and a maximum number of iterations to stop
    iterating, performs Gauss-Seidel iteration scheme on system to approximate a solution
    until a stop condition is reached. Minimum threshold, maximum number of iterations, and initial
    guess can be specified but default to 10^-5, 30, and the zero vector respectively. The final
    approximation reached is returned.
    :param input_matrix: Matrix representing system to solve.
    :param b: Vector of system outputs.
    :param x0: Initial guess for solution.
    :param threshold: Minimum distance between two vector approximations for iteration to stop.
    :param steps: Maximum number of iterations before stopping.
    :return: Vector of final iteration output.
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    lower = np.tril(input_matrix, -1)
    upper = np.triu(input_matrix, 1)
    diagonal = np.diagflat(np.diagonal(input_matrix))

    ld_inverse = np.linalg.inv(lower + diagonal)
    iter_mat = -np.matmul(ld_inverse, upper)
    c = np.matmul(ld_inverse, b)

    final = iterate_matrix(iter_mat, c, x0=x0, threshold=threshold, steps=steps)

    conclude(input_matrix, b, final)
    return final


def sor_iteration(input_matrix: np.array, b: np.array, x0: np.array = None, threshold: float = 10 ** (-5),
                  steps=30, w=1.5) -> np.array:
    """
    Given a matrix representing the system to be solved, a b vector representing
    the output vector of the system, an initial guess, given minimum threshold
    for consecutive approximations to stop iterating, a maximum number of iterations to stop
    iterating, and a Relaxation parameter, performs Successive Over-Relaxation iteration scheme
    on system to approximate
    a solution until a stop condition is reached. Minimum threshold, maximum number of iterations,
    initial guess, and relaxation parameter can be specified but default to 10^-5, 30,
    the zero vector, and 1.5 respectively.
    The final approximation reached is returned.
    :param input_matrix: Matrix representing system to solve.
    :param b: Vector of system outputs.
    :param x0: Initial guess for solution.
    :param threshold: Minimum distance between two vector approximations for iteration to stop.
    :param steps: Maximum number of iterations before stopping.
    :param w: Relaxation parameter.
    :return: Vector of final iteration output.
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    lower = np.tril(input_matrix, -1)
    upper = np.triu(input_matrix, 1)
    diagonal = np.diagflat(np.diagonal(input_matrix))

    weighted_ld_inverse = np.linalg.inv(w * lower + diagonal)
    iter_mat = -np.matmul(weighted_ld_inverse, w * upper + (w - 1) * diagonal)
    c = np.matmul(weighted_ld_inverse, w * b)

    final = iterate_matrix(iter_mat, c, x0=x0, threshold=threshold, steps=steps)
    conclude(input_matrix, b, final)
    return final


a = np.array([[2, 0], [1, 2]])
bb = np.array([6, 6]).transpose()
print(sor_iteration(a, bb))
print(is_diagonally_dominent(a))