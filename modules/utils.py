import numpy as np
from scipy.linalg import solve_discrete_are

def calc_P(A, B, Q, R):
    """ Solve DARE """
    return solve_discrete_are(A, B, Q, R)

def calc_K(A, B, Q, R):
    """  Get infinite-horizon LQR gain """
    P = calc_P(A, B, Q, R)
    BP = B.T @ P
    LHS = -(R + (BP @ B))
    RHS = BP @ A
    K = np.linalg.solve(LHS, RHS)
    return K