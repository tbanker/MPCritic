import os
import sys
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

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout