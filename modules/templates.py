import numpy as np
import torch
import casadi as ca
import sys
import os
import do_mpc

from typing import Optional
import gymnasium as gym

kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}

def template_model(f, n, m, symvar_type='MX'):
    """ template_model: Variables / RHS / AUX """
    # Following the construction in https://www.do-mpc.com/en/latest/example_gallery/oscillating_masses_discrete.html
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)
    
    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(n,1)) # (1,n) to align with torch

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(m,1))

    # Define difference equation
    _z = ca.vertcat(_x, _u)
    x_next = f(_z.T).T
    model.set_rhs('x', x_next)

    # Build the model
    model.setup()

    return model


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
    # model.set_expression(expr_name='stage_cost', expr=ca.sum1(_x**2)+ca.sum1(_u**2))
    # model.set_expression(expr_name='terminal_cost', expr=ca.sum1(_x**2))
   
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
            A=0.5 * np.array([[1., 0., 2., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 2.],
                              [1., 0., 0., 1.],], np.float32),
            B=np.array([[0.5, 0],
                        [0., 0],
                        [0., 0],
                        [0., 0.5]], np.float32),
            Q=np.diag(np.ones(4, np.float32)),
            R=np.diag(np.ones(2, np.float32)),
            max_timesteps=10
        ):

        n = A.shape[0]
        m = B.shape[1]
        assert (A.shape == (n,n)) and (B.shape == (n,m))
        assert (Q.shape == (n,n)) and (R.shape == (m,m))
        self.n = n
        self.m = m

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.max_timesteps = max_timesteps
        self.current_step = 0

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
        truncated = (self.current_step >= self.max_timesteps)
        return {'final_observation' : self._state, "TimeLimit.truncated": truncated}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """ Reset the system """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.current_step = 0

        # Choose the state uniformly at random
        self._state = self.np_random.uniform(low=-1., high=1., size=(self.n)).astype(np.float32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """ Advance system """
        self.current_step += 1

        reward = -(self._state @ self.Q @ self._state.T + action @ self.R @ action.T)

        self._state = self.A @ self._state.T + self.B @ action.T
        observation = self._get_obs()
        
        info = self._get_info()
        terminated = False
        truncated = info["TimeLimit.truncated"]

        return observation, reward, terminated, truncated, info
    
if __name__ == "__main__":
    import os
    import random
    import time
    from dataclasses import dataclass
    import tyro

    from stable_baselines3.common.buffers import ReplayBuffer

    from mpcritic import InputConcat
    from mpcomponents import LinearDynamics, QuadraticStageCost
    from templates import LQREnv
    from utils import fill_rb

    """ CleanRL setup """
    gym.register(
        id="gymnasium_env/LQR-v0",
        entry_point=LQREnv,
    )

    @dataclass
    class Args:
        exp_name: str = os.path.basename(__file__)[: -len(".py")]
        """the name of this experiment"""
        seed: int = 1
        """seed of the experiment"""
        torch_deterministic: bool = True
        """if toggled, `torch.backends.cudnn.deterministic=False`"""
        cuda: bool = True
        """if toggled, cuda will be enabled by default"""
        track: bool = False
        """if toggled, this experiment will be tracked with Weights and Biases"""
        wandb_project_name: str = "cleanRL"
        """the wandb's project name"""
        wandb_entity: str = None
        """the entity (team) of wandb's project"""
        capture_video: bool = False
        """whether to capture videos of the agent performances (check out `videos` folder)"""
        save_model: bool = False
        """whether to save model into the `runs/{run_name}` folder"""
        upload_model: bool = False
        """whether to upload the saved model to huggingface"""
        hf_entity: str = ""
        """the user or org name of the model repository from the Hugging Face Hub"""

        # Algorithm specific arguments
        env_id: str = "gymnasium_env/LQR-v0" # "Hopper-v4"
        """the environment id of the Atari game"""
        total_timesteps: int = 10000
        """total timesteps of the experiments"""
        learning_rate: float = 3e-4
        """the learning rate of the optimizer"""
        buffer_size: int = int(1e6)
        """the replay memory buffer size"""
        gamma: float = 0.99
        """the discount factor gamma"""
        tau: float = 0.005
        """target smoothing coefficient (default: 0.005)"""
        batch_size: int = 64
        """the batch size of sample from the reply memory"""
        exploration_noise: float = 0.1
        """the scale of exploration noise"""
        learning_starts: int = 25e3
        """timestep to start learning"""
        policy_frequency: int = 2
        """the frequency of training policy (delayed)"""
        noise_clip: float = 0.5
        """noise clip parameter of the Target Policy Smoothing Regularization"""

    def make_env(env_id, seed, idx, capture_video, run_name):
        def thunk():
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            return env

        return thunk

    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    n_envs = 1
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name) for _ in range(n_envs)])

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        kwargs['device'],
        handle_timeout_termination=False,
        n_envs=n_envs
    )

    """ System information """
    b, n, m = envs.num_envs, envs.get_attr("n")[0], envs.get_attr("m")[0]
    A, B = envs.get_attr("A")[0], envs.get_attr("B")[0]
    Q, R = envs.get_attr("Q")[0], envs.get_attr("R")[0]

    f = LinearDynamics(n, m, A, B)
    concat_f = InputConcat(f)
    l = QuadraticStageCost(n, m, Q, R)
    concat_l = InputConcat(l)

    """ Fill replay buffer """
    obs, _ = envs.reset(seed=args.seed)
    obs = fill_rb(rb, envs, obs, n_samples=64)

    """ Verify predictions and rewards match """
    s = torch.from_numpy(rb.observations[:64]).squeeze(1)
    a = torch.from_numpy(rb.actions[:64]).squeeze(1)
    r = torch.from_numpy(rb.rewards[:64])
    s_next = torch.from_numpy(rb.next_observations[:64]).squeeze(1)
    print(torch.allclose(s_next, concat_f(s, a)))
    print(torch.allclose(r, -concat_l(s, a)))