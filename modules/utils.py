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

def fill_rb(rb, envs, obs, policy=None, n_transitions=1000):
    """ Add samples to RB following Clean-RL  """
    for _ in range(n_transitions):
        if policy is None:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
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