import numpy as np
from scipy.linalg import solve_discrete_are
import torch

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

def stable(A):
    Re_eigvals_A = np.linalg.eigvals(A).real
    if np.all(np.abs(Re_eigvals_A) < 1):
        print("Open-loop Asymptotic stable")
    elif np.all(np.abs(Re_eigvals_A) <= 1) and np.any(Re_eigvals_A == 1):
        print("Open-loop Lyapunov stable")
    else:
        print("Open-loop Unstable")

def controllable(A, B):
    n = A.shape[0]
    Re_eigvals_A = np.linalg.eigvals(A).real

    eig_rank = np.zeros_like(Re_eigvals_A)
    for i, eigval in enumerate(Re_eigvals_A):
        M = np.concatenate([eigval * np.eye(n) - A, B], axis=1)
        eig_rank[i] = np.linalg.matrix_rank(M)
    if np.all(eig_rank == n):
        print("Controllable")

    else:
        ge1_eigvals_A = Re_eigvals_A[np.abs(Re_eigvals_A) >= 1.]
        eig_rank = np.zeros_like(ge1_eigvals_A)
        for i, eigval in enumerate(ge1_eigvals_A):
            M = np.concatenate([eigval * np.eye(n) - A, B], axis=1)
            eig_rank[i] = np.linalg.matrix_rank(M)
        if np.all(eig_rank == n):
            print("Stabilizable")

def fill_rb(rb, envs, obs, policy=None, sampling="Normal", n_samples=1000):
    """ Add samples to RB following Clean-RL  """
    for _ in range(n_samples):
        if sampling == "Normal":
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)]) # normal dist. N(0,1^2)
        elif sampling == "Uniform":
            actions = np.random.uniform(-1,1, size=(envs.num_envs, envs.single_action_space.shape[0])).astype(np.float32)
        else:
            actions = policy(torch.from_numpy(obs)).detach().numpy()
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

    return obs