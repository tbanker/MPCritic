import numpy as np
import casadi as ca
from casadi import tools
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

from typing import Optional
import numpy as np
import gymnasium as gym

def template_linear_model(n, m, symvar_type='MX'):
    """ template_model: Variables / RHS / AUX """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(n,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(m,1))

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    model.set_expression(expr_name='stage_cost', expr=ca.sum1(_x**2)+ca.sum1(_u**2))
    model.set_expression(expr_name='terminal_cost', expr=ca.sum1(_x**2))
   
    # Fixed parameters:
    A = model.set_variable('_p', var_name='A', shape=(n,n))
    B = model.set_variable('_p', var_name='B', shape=(n,m))

    x_next = model.set_variable(var_type='_z', var_name='x_next', shape=(n,1))

    model.set_rhs('x', x_next)

    model.set_alg('x_next', x_next-A@_x-B@_u)

    model.setup()

    return model

def template_linear_simulator(model, sim_p):
    """ template_optimizer: tuning parameters """
    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 1.0)

    p_num = simulator.get_p_template()
    for name, value in sim_p.items():
        p_num[name] = value

    def p_fun(t_now):
        return p_num

    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator

class LQREnv(gym.Env):
    def __init__(self,
            n=2,
            m=2,
            A=np.diag(np.ones(2, np.float32)),
            B=np.diag(np.ones(2, np.float32)),
            Q=np.diag(np.ones(2, np.float32)),
            R=np.diag(np.ones(2, np.float32)),
        ):
        
        assert (A.shape == (n,n)) and (B.shape == (n,m))
        assert (Q.shape == (n,n)) and (R.shape == (m,m))
        self.n = n
        self.m = m

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self._state = np.ones((n), dtype=np.int32)

        self.observation_space = gym.spaces.Box(
            low=np.full((n), -np.inf).astype(np.float32), high=np.full((self.n), np.inf).astype(np.float32),
            shape=(self.n,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.full((m), -np.inf).astype(np.float32), high=np.full((m), np.inf).astype(np.float32),
            shape=(m,), dtype=np.float32
        )
        
    def _get_obs(self):
        return self._state
    
    def _get_info(self):
        return {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the state uniformly at random
        self._state = self.np_random.uniform(low=-3, high=3, size=(self.n)).astype(np.float32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        self._state = self.A @ self._state.T + self.B @ action.T

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._state, np.zeros(n, dtype=np.float32))
        truncated = False
        reward = -(self._state @ self.Q @ self._state.T + action @ self.R @ action.T)
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
if __name__ == "__main__":

    np_kwargs = {'dtype' : np.float32}

    gym.register(
        id="gymnasium_env/LQR-v0",
        entry_point=LQREnv,
    )

    """ Testing Environments """
    gym.make("gymnasium_env/LQR-v0")
    gym.make("gymnasium_env/LQR-v0", max_episode_steps=10)
    gym.make_vec("gymnasium_env/LQR-v0", num_envs=3)

    """ Testing dynamics """
    n = 3
    m = n
    Q, R = np.diag(np.ones(n, **np_kwargs)), np.diag(np.ones(m, **np_kwargs))
    A = np.diag(np.ones(n, **np_kwargs))
    B = np.diag(np.ones(m, **np_kwargs))
    env_kwargs = {k:v for k,v in zip(["n", "m", "Q", "R", "A", "B"], [n, m, Q, R, A, B])}

    env = gym.make("gymnasium_env/LQR-v0", **env_kwargs)
    K = -0.5 * np.diag(np.ones(n, **np_kwargs))

    obs, _ = env.reset()
    # action = K @ obs
    action = np.zeros(m, **np_kwargs)
    next_obs, reward, _, _, _ = env.step(action)
    print(obs, action, reward, next_obs)