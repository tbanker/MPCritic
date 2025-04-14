import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c

import do_mpc
import casadi as ca
from copy import copy

from neuromancer.dataset import DictDataset
from neuromancer.system import Node, System
from neuromancer.problem import Problem
import sys
sys.path.append('')
from modules.dpcontrol import DPControl

np_kwargs = {'dtype' : np.float32}
kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}

class InputConcat(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, u):
        """ f(x,u) -> f(z) """
        # L4CasADi models must have 1 input (z), but Neuromancer needs 2 (x,u)
        z = torch.concat((x,u), dim=-1)
        return self.module(z)

def template_model(f, n, m, symvar_type='MX'):
    """ template_model: Variables / RHS / AUX """
    # Following the construction in https://www.do-mpc.com/en/latest/example_gallery/oscillating_masses_discrete.html
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)
    
    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(n,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(m,1))

    # Define difference equation
    _z = ca.vertcat(_x, _u)
    x_next = f(_z.T).T # to align with torch model
    model.set_rhs('x', x_next)

    # Build the model
    model.setup()

    return model

class MPCritic(nn.Module):
    def __init__(self, dpcontrol, unc_p=None, mpc_settings=None, terminal_Q=False):
        super().__init__()

        self.dpcontrol = dpcontrol

        self.critic_parameters = nn.ParameterDict({'l': self.dpcontrol.l.module, 'V': self.dpcontrol.V})

        # Configure network
        self.H = self.dpcontrol.H # mpc horizon per do-mpc
        self.dx_node = Node(self.dpcontrol.dynamics.dx, ['x', 'u'], ['x_next'], name='dynamics_model')
        self.goal_node = Node(self.dpcontrol.goal_map, ['x'], ['z'])
        self.mu_node = Node(self.dpcontrol.mu, ['x_next'], ['u'], name='policy')
        self.x_shift = Node(lambda x: x, ['x_next'], ['x'], name='x_shift')
        self.l_node = self.dpcontrol.l_node
        self.V_node = self.dpcontrol.V_node
        self.model = System([self.goal_node, self.dx_node, self.mu_node, self.x_shift, self.l_node, self.V_node], nsteps=self.H + 2)
        self.model_kwargs = {'dtype' : list(self.model.parameters())[0].dtype,
                             'device' : list(self.model.parameters())[0].device,}

        # Formulate problem
        self.obj = dpcontrol.obj
        self.problem = Problem([self.model], self.obj)

        # Critic evaluation functions
        self.batched_fwd_s = torch.vmap(self.forward_critic_s)
        self.batched_fwd_sa = torch.vmap(self.forward_critic_sa)

        # MPC settings & objects
        self.mpc_settings = {'n_horizon': self.dpcontrol.H,
                             'n_robust': 0,
                             'open_loop': False,
                             't_step': 1.0,
                             'store_full_solution': True,
                             'store_lagr_multiplier' : True,
                             'nlpsol_opts': {'ipopt.linear_solver': 'mumps'},
                             } if mpc_settings == None else mpc_settings
        self.dx_mpc = self.dpcontrol.dynamics.dx.module
        self.l_mpc = self.dpcontrol.l.module
        self.V_mpc = self.dpcontrol.V
        self.unc_p = unc_p

        # L4C settings & objects
        self.l4c_kwargs = {'device' : 'cpu',
                           'batched' : True,
                           'mutable' : True,
                           'generate_jac' : True,
                           'generate_jac_jac' : True,
                           'generate_jac_adj1' : True,
                           'generate_adj1' : False,}
        self.dx_l4c = l4c.L4CasADi(self.dx_mpc, **self.l4c_kwargs)
        self.l_l4c = l4c.L4CasADi(self.l_mpc, **self.l4c_kwargs)
        self.V_l4c = l4c.L4CasADi(self.V_mpc, **self.l4c_kwargs)

    def forward(self, s, a=None):
        """ batched critic operation """
        if a != None:
            return self.batched_fwd_sa(s, a)
        else:
            return self.batched_fwd_s(s)
    
    def forward_critic_s(self, s):
        """ single-batched critic(s,\mu(s)) operation """
        x = self.x_shift({'x_next' : s})
        u_pi = self.mu_node({'x_next' : s})
        input_dict = DictDataset({'x' : x['x'].view(1,1,-1),
                                  'u' : u_pi['u'].view(1,1,-1)})
        input_dict.datadict['name'] = 'eval'

        output_dict = self.problem(input_dict.datadict)
        return -output_dict['eval_loss'].view(1)
    
    def forward_critic_sa(self, s, a):
        """ single-batched critic(s,a) operation """
        input_dict = DictDataset({'x' : s.view(1,1,-1),
                                  'u' : a.view(1,1,-1)})
        input_dict.datadict['name'] = 'eval'
        # print(s)
        output_dict = self.problem(input_dict.datadict)
        return -output_dict['eval_loss'].view(1)
    
    def forward_mpc(self, x0):
        """ single-batched MPC operation """
        return self.mpc.make_step(x0)
    
    def _update_dynamics(self, kwargs={}):
        self.dpcontrol.dynamics.train(**kwargs)
        return
    
    def _update_controller(self, kwargs={}):
        self.dpcontrol.train(**kwargs)
        return

    def train_f_mu(self, train_f=True, train_mu=True, f_kwargs={}, mu_kwargs={}):
        if train_f:
            self._update_dynamics(kwargs=f_kwargs)
        if train_mu:
            self._update_controller(kwargs=mu_kwargs)
        return

    def l4c_update(self):
        """ update all L4CasADi objects (do prior to recalling setup_mpc) """
        self.dx_l4c = l4c.L4CasADi(self.dx_mpc, **self.l4c_kwargs)
        self.l_l4c = l4c.L4CasADi(self.l_mpc, **self.l4c_kwargs)
        self.V_l4c = l4c.L4CasADi(self.V_mpc, **self.l4c_kwargs)
    
    def setup_mpc(self):
        """ setup MPC problem """

        # Convert torch model to do-mpc-friendly model
        self.dx_dompc = template_model(self.dx_l4c, self.dpcontrol.nx, self.dpcontrol.nu) 

        # Instantiate MPC
        mpc = do_mpc.controller.MPC(self.dx_dompc)
        mpc.settings.__dict__.update(**self.mpc_settings)
        mpc.settings.supress_ipopt_output() # please be quiet

        z = ca.transpose(ca.vertcat(self.dx_dompc._x, self.dx_dompc._u))
        lterm = self.l_l4c.forward(z)
        x = ca.transpose(self.dx_dompc._x)
        mterm = self.V_l4c.forward(x)
        # forward to 'build' l4c_model, required before L4CasADi.update()
        self.dx_l4c(z)

        mpc.set_objective(lterm=lterm, mterm=mterm)
        mpc.set_rterm(u=0.)

        if self.dpcontrol.xlim is not None:
            mpc.bounds['lower','_x','x'] = self.dpcontrol.xlim[0]
            mpc.bounds['upper','_x','x'] = self.dpcontrol.xlim[1]
        if self.dpcontrol.ulim is not None:
            mpc.bounds['lower','_u','u'] = self.dpcontrol.ulim[0]
            mpc.bounds['upper','_u','u'] = self.dpcontrol.ulim[1]

        mpc.setup()

        self.mpc = mpc

    def init_mpc(self, x0):
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    def online_mpc_update(self, x0, full=True):
        """ update MPC online, copying old MPC data into new MPC """
        mpc_data, mpc_t0 = copy(self.mpc.data), copy(self.mpc._t0)
        if full == True:
            self.l4c_update() # update l4c_lterm, mterm, model
        self.setup_mpc()
        self.init_mpc(x0)

        self.mpc.data, self.mpc._t0 = mpc_data, mpc_t0

    def _mpc_state(self, x0):
        return np.float32(np.reshape(x0, self.dpcontrol.env.observation_space.shape[::-1]))
    
    def _rl_action(self, u0):
        return torch.from_numpy(np.float32(np.reshape(u0, self.dpcontrol.env.action_space.shape)))

if __name__ == '__main__':
    import os
    import random
    import time
    from dataclasses import dataclass
    import tyro

    import numpy as np
    import gymnasium as gym
    from stable_baselines3.common.buffers import ReplayBuffer

    from dynamics import Dynamics
    from dpcontrol import DPControl
    from mpcomponents import QuadraticStageCost, QuadraticTerminalCost, LinearDynamics, LinearPolicy, GoalMap
    from utils import calc_K, calc_P

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
        wandb_project_name: str = "mpcritic-dev"
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
        env_id: str = "lqr-v0"
        """the environment id of the Atari game"""
        total_timesteps: int = 10000
        """total timesteps of the experiments"""
        learning_rate: float = 3e-4
        """the learning rate of the optimizer"""
        num_envs: int = 1
        """the number of parallel game environments"""
        buffer_size: int = int(1e3)
        """the replay memory buffer size"""
        gamma: float = 0.99
        """the discount factor gamma"""
        tau: float = 0.005
        """target smoothing coefficient (default: 0.005)"""
        batch_size: int = 256
        """the batch size of sample from the reply memory"""
        policy_noise: float = 0.2
        """the scale of policy noise"""
        exploration_noise: float = 0.1
        """the scale of exploration noise"""
        learning_starts: int = 25e3
        """timestep to start learning"""
        policy_frequency: int = 2
        """the frequency of training policy (delayed)"""
        noise_clip: float = 0.5
        """noise clip parameter of the Target Policy Smoothing Regularization"""

        # LQR specific arguments
        n: int = 4
        """state AND action dimension"""

    def make_env(env_id, seed, idx, capture_video, run_name, path, goal_map):
        if "lqr" in env_id:
            from envs.LQR.template_model import template_model
            from envs.LQR.template_mpc import template_mpc
            from envs.LQR.template_simulator import template_simulator
            from envs.DoMPCEnv import DoMPCEnv

            gym.register(
            id=env_id,
            entry_point=DoMPCEnv,
                )  

            model = template_model(n=args.n, m=args.n)
            max_x = np.ones(args.n).flatten()
            min_x = -np.ones(args.n).flatten() # writing like this to emphasize do-mpc sizing convention
            max_u = np.ones(args.n).flatten()
            min_u = -np.ones(args.n).flatten()
            bounds = {'x_low' : min_x, 'x_high' : max_x, 'u_low' : min_u, 'u_high' : max_u}
            goal_map = goal_map
            num_steps = 50
            kwargs = {'disable_env_checker': True, 'template_simulator': template_simulator, 'model': model,
                    'num_steps': num_steps, 'bounds': bounds, 'same_state': None,
                    'goal_map': goal_map, 'smooth_reward': False, 'sa_reward': True,
                    'path': path}

        else:
            kwargs = {}
    
        def thunk():
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id, **kwargs)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            return env

        return thunk

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    goal_map = GoalMap()
    exp_path = f"runs/{run_name}/"
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name, exp_path, goal_map) for i in range(args.num_envs)]
    )

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=kwargs['device'],
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    """ System information """
    obs, _ = envs.reset(seed=args.seed)
    b, n, m = envs.num_envs, envs.get_attr("n")[0], envs.get_attr("m")[0]
    A_env, B_env = envs.envs[0].simulator.p_fun(0)['A'].full().astype(np_kwargs['dtype']), envs.envs[0].simulator.p_fun(0)['B'].full().astype(np_kwargs['dtype'])
    Q, R = np.diag(np.ones(n)).astype(np_kwargs['dtype']), envs.envs[0].sa_reward_scale*np.diag(np.ones(m)).astype(np_kwargs['dtype'])

    P_opt = calc_P(A_env, B_env, Q, R).astype(np_kwargs['dtype'])

    """ Model setup """
    K = -0.5 * np.ones((m,n), **np_kwargs)

    mpc_horizon = 1
    l = QuadraticStageCost(n, m, Q, R)
    V = QuadraticTerminalCost(n, P_opt)
    f = LinearDynamics(n, m, A_env, B_env)
    mu = LinearPolicy(n, m, K)

    concat_f = InputConcat(f)
    dynamics = Dynamics(envs, rb, dx=concat_f)

    xlim = np.vstack([-np.ones(n), np.ones(n)])
    ulim = np.vstack([-np.ones(m), np.ones(m)])
    dpcontrol = DPControl(envs, rb=rb, H=mpc_horizon, dynamics=dynamics, l=l, V=V, mu=mu, goal_map=GoalMap(), xlim=xlim, ulim=ulim, loss='penalty', scale=10.)

    critic = MPCritic(dpcontrol)

    """ Different outputs for given action """
    obs, _ = envs.reset()
    x = torch.from_numpy(obs)

    q_s = critic(s=x)
    u = torch.zeros((b,m), **kwargs)
    q_sa = critic(s=x, a=u)
    print(f'Q(s) != Q(s,a=0): {not torch.allclose(q_s, q_sa)}')

    """ Same outputs when action given/not given """
    q_s = critic(s=x)
    q_sa = critic(s=x, a=mu(x))
    print(f'Q(s) == Q(s,a=\mu(s)): {torch.allclose(q_s, q_sa)}')

    """ Same outputs for optimal critic Q^* and value function V^* """
    K = calc_K(A_env, B_env, Q, R).astype(np_kwargs['dtype'])
    mu = LinearPolicy(n, m, K)
    dpcontrol = DPControl(envs, rb=rb, H=mpc_horizon, dynamics=dynamics, l=l, V=V, mu=mu, goal_map=GoalMap(), xlim=xlim, ulim=ulim, loss='penalty', scale=10.)
    critic = MPCritic(dpcontrol)

    q_s = critic(s=x)
    q_sa = critic(s=x, a=mu(x))
    P_opt = torch.from_numpy(P_opt)
    V_opt = -(x @ P_opt * x).sum(axis=1, keepdims=True).to(dtype=q_s.dtype)
    print(f'Q^*(s) == Q^*(s,a=K^*(s))): {torch.allclose(q_s, q_sa)}')
    print(f'Q^*(s) == V^*(s)): {torch.allclose(q_s, V_opt)}')