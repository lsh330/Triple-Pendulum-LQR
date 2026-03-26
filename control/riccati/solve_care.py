from scipy.linalg import solve_continuous_are


def solve_riccati(A, B, Q, R):
    """Solve the continuous algebraic Riccati equation.

    Returns the solution matrix P.
    """
    P = solve_continuous_are(A, B, Q, R)
    return P
