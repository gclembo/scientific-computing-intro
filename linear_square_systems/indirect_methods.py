import numpy as np


def iterate_matrix(input_matrix: np.array, b: np.array, x0: np.array = None,
                   threshold: float = 1e-5, iterations: int = 30) -> np.array:
    """
    Given a matrix to multiply current approximation by in iteration scheme, a vector to add in
    iteration scheme, an initial guess,a given minimum threshold for consecutive approximations
    to stop iterating, and a maximum number of iterations to stop iterating, multiplies initial
    guess by iteration matrix and adds b vector each iteration until a stop condition is reached.
    The minimum threshold, maximum number of iterations, and initial guess can be specified but
    default to 1e-5, 30, and the zero vector respectively.

    Parameters
    ----------
    input_matrix : np.array
        Matrix to multiply current approximation by in iteration.
    b : np.array
        Vector to add in iteration scheme.
    x0 : np.array, optional
        Initial guess for iteration. Defaults to the zero vector.
    threshold: float, default 1e-5
        Minimum distance between two vector approximations for iteration to stop. Defaults to 1e-5.
    iterations: int, default 30
        Maximum number of iterations before stopping. Defaults to 30.

    Returns
    -------
    np.array
        Vector of final iteration output.
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    iteration = 1
    x1 = np.matmul(input_matrix, x0) + b
    dist = np.linalg.norm(x1 - x0)

    while iteration < iterations and dist >= threshold:
        x0 = x1
        x1 = np.matmul(input_matrix, x0) + b
        dist = np.linalg.norm(x1 - x0)
        iteration += 1

    if iteration < iterations:
        print("Iteration stopped after successive solutions were closer than " + str(threshold))
        print("This took " + str(iteration) + " iterations")
    else:
        print("Iteration stopped after " + str(iterations) + " iterations")

    return x1


def is_diagonally_dominant(input_matrix: np.array) -> bool:
    """
    Given a matrix, returns if the matrix is strictly diagonally dominant.

    Parameters
    ----------
    input_matrix : np.array
        Matrix to evaluate.

    Returns
    -------
    bool
        If given matrix is strictly diagonally dominant.
    """
    for i in range(input_matrix.shape[0]):
        if 2 * abs(input_matrix[i][i]) <= sum(abs(input_matrix[i])):
            return False
    return True


def conclude(input_matrix: np.array, b: np.array, final: np.array) -> None:
    """
    Given a matrix representing the system that was attempted to be solved, a b vector of
    dependent values for solving the system, and a final approximation of the solution to the system,
    prints conclusion for the user to interpret.

    Parameters
    ----------
    input_matrix : np.array
        Matrix representing system that was attempted to be solved.
    b : np.array
        Vector of dependent values for solving the system.
    final : np.array
        Final approximation of system solution.

    Returns
    -------
    None
    """
    predicted = np.matmul(input_matrix, final)
    output_dist = np.linalg.norm(predicted - b)
    print("The vector arrived at: " + str(final) + " predicts a b vector of " + str(predicted))
    print("The distance between actual: " + str(b) + " and predicted is " + str(output_dist))


def neumann_iteration(input_matrix: np.array, b: np.array, x0: np.array = None,
                      threshold: float = 1e-5, iterations: int = 30) -> np.array:
    """
    Given a matrix representing the system to be solved, a b vector of dependent values for
    solving the system, an initial guess, a given minimum threshold for consecutive approximations
    to stop iterating, and a maximum number of iterations to stop iterating, performs
    Neumann iteration scheme on system to approximate a solution
    until a stop condition is reached. The minimum threshold, maximum number of iterations,
    and initial guess can be specified but default to 1e-5, 30, and the zero vector respectively.
    The final approximation reached is returned.

    Parameters
    ----------
    input_matrix : np.array
        Matrix representing system to solve.
    b : np.array
        Vector of dependent values for solving the system.
    x0 : np.array, optional
        Initial guess for solution. Defaults to the zero vector.
    threshold : float, default 1e-5
        Minimum distance between two vector approximations for iteration to stop. Defaults to 1e-5.
    iterations : int, default 30
        Maximum number of iterations before stopping. Defaults to 30.

    Returns
    -------
    np.array
        Vector of final iteration output.
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    iter_mat = np.eye(input_matrix.shape[0]) - input_matrix
    final = iterate_matrix(iter_mat, b, x0=x0, threshold=threshold, iterations=iterations)
    conclude(input_matrix, b, final)
    return final


def jacobi_iteration(input_matrix: np.array, b: np.array, x0: np.array = None,
                     threshold: float = 1e-5, iterations: int = 30) -> np.array:
    """
    Given a matrix representing the system to be solved, a b vector of dependent values for
    solving the system, an initial guess, a given minimum threshold for consecutive approximations
    to stop iterating, and a maximum number of iterations to stop iterating, performs
    Jacobi iteration scheme on system to approximate a solution
    until a stop condition is reached. The minimum threshold, maximum number of iterations,
    and initial guess can be specified but default to 1e-5, 30, and the zero vector respectively.
    The final approximation reached is returned.

    Parameters
    ----------
    input_matrix : np.array
        Matrix representing system to solve.
    b : np.array
        Vector of dependent values for solving the system.
    x0 : np.array, optional
        Initial guess for solution. Defaults to the zero vector.
    threshold : float, default 1e-5
        Minimum distance between two vector approximations for iteration to stop. Defaults to 1e-5.
    iterations : int, default 30
        Maximum number of iterations before stopping. Defaults to 30.

    Returns
    -------
    np.array
        Vector of final iteration output.
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    lower = np.tril(input_matrix, -1)
    upper = np.triu(input_matrix, 1)
    diagonal = np.diagflat(np.diagonal(input_matrix))
    diag_inverse = np.linalg.inv(diagonal)
    iter_mat = -np.matmul(diag_inverse, (lower + upper))
    c = np.matmul(diag_inverse, b)

    final = iterate_matrix(iter_mat, c, x0=x0, threshold=threshold, iterations=iterations)
    conclude(input_matrix, b, final)
    return final


def gauss_seidel_iteration(input_matrix: np.array, b: np.array, x0: np.array = None,
                           threshold: float = 1e-5, iterations: int = 30) -> np.array:
    """
    Given a matrix representing the system to be solved, a b vector of dependent values for
    solving the system, an initial guess, a given minimum threshold for consecutive approximations
    to stop iterating, and a maximum number of iterations to stop iterating, performs
    Gauss-Seidel iteration scheme on system to approximate a solution
    until a stop condition is reached. The minimum threshold, maximum number of iterations,
    and initial guess can be specified but default to 1e-5, 30, and the zero vector respectively.
    The final approximation reached is returned.

    Parameters
    ----------
    input_matrix : np.array
        Matrix representing system to solve.
    b : np.array
        Vector of dependent values for solving the system.
    x0 : np.array, optional
        Initial guess for solution. Defaults to the zero vector.
    threshold : float, default 1e-5
        Minimum distance between two vector approximations for iteration to stop. Defaults to 1e-5.
    iterations : int, default 30
        Maximum number of iterations before stopping. Defaults to 30.

    Returns
    -------
    np.array
        Vector of final iteration output.
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    lower = np.tril(input_matrix, -1)
    upper = np.triu(input_matrix, 1)
    diagonal = np.diagflat(np.diagonal(input_matrix))

    ld_inverse = np.linalg.inv(lower + diagonal)
    iter_mat = -np.matmul(ld_inverse, upper)
    c = np.matmul(ld_inverse, b)

    final = iterate_matrix(iter_mat, c, x0=x0, threshold=threshold, iterations=iterations)

    conclude(input_matrix, b, final)
    return final


def sor_iteration(input_matrix: np.array, b: np.array, x0: np.array = None,
                  threshold: float = 1e-5, iterations: int = 30, w: float = 1.5) -> np.array:
    """
    Given a matrix representing the system to be solved, a b vector of dependent values for
    solving the system, an initial guess, a given minimum threshold
    for consecutive approximations to stop iterating, a maximum number of iterations to stop
    iterating, and a relaxation parameter, performs Successive Over-Relaxation iteration scheme
    on system to approximate a solution until a stop condition is reached. The minimum threshold,
    maximum number of iterations, initial guess, and relaxation parameter can be specified but
    default to 1e-5, 30, the zero vector, and 1.5 respectively.
    The final approximation reached is returned.

    Parameters
    ----------
    input_matrix : np.array
        Matrix representing system to solve.
    b : np.array
        Vector of dependent values for solving the system.
    x0 : np.array, optional
        Initial guess for solution. Defaults to the zero vector.
    threshold : float, default 1e-5
        Minimum distance between two vector approximations for iteration to stop. Defaults to 1e-5.
    iterations : int, default 30
        Maximum number of iterations before stopping. Defaults to 30.
    w : float, default 1.5
        Relaxation parameter. Defaults to 1.5.

    Returns
    -------
    np.array
        Vector of final iteration output.
    """
    if x0 is None:
        x0 = np.zeros(b.shape)
    lower = np.tril(input_matrix, -1)
    upper = np.triu(input_matrix, 1)
    diagonal = np.diagflat(np.diagonal(input_matrix))

    weighted_ld_inverse = np.linalg.inv(w * lower + diagonal)
    iter_mat = -np.matmul(weighted_ld_inverse, w * upper + (w - 1) * diagonal)
    c = np.matmul(weighted_ld_inverse, w * b)

    final = iterate_matrix(iter_mat, c, x0=x0, threshold=threshold, iterations=iterations)
    conclude(input_matrix, b, final)
    return final
