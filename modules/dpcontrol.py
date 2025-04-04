import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer

# neuromancer stuff
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dynamics import integrators
from neuromancer.constraint import variable, Objective
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss, AugmentedLagrangeLoss
from neuromancer.dataset import DictDataset
from neuromancer.trainer import Trainer
from neuromancer.psl import signals
import torch.optim as optim
from torch.utils.data import DataLoader

# MPCritic stuff
import sys
sys.path.append('')
from modules.mpcomponents import QuadraticStageCost, PDQuadraticStageCost, QuadraticTerminalCost, PDQuadraticTerminalCost, LinearDynamics, LinearPolicy, GoalMap
from modules.dynamics import Dynamics


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

class DPControl(nn.Module):
    def __init__(self, env, H=5, rb=None, dynamics=None, l=None, V=None, mu=None, goal_map=None, linear_dynamics=False, xlim=None, ulim=None, loss=None, scale=10.0, opt="AdamW", lr=0.001):
        super().__init__()

        self.env = env
        self.rb = rb if rb != None else ReplayBuffer(int(1e6),
                                                env.single_observation_space,
                                                env.single_action_space,
                                                device=kwargs['device'],
                                                handle_timeout_termination=False,
                                                n_envs=env.num_envs
                                            )

        # Configure network
        self.nx = np.array(env.single_observation_space.shape).prod()
        self.nu = np.array(env.single_action_space.shape).prod()
        self.H = H # mpc horizon per do-mpc
        if goal_map is not None:
            self.goal_map = goal_map if goal_map != None else GoalMap()
        self.ny = self.goal_map.ny if self.goal_map.ny != None else self.nx
        self.mu = mu if mu != None else blocks.MLP(self.nx, self.nu, bias=True,
                                                      linear_map=torch.nn.Linear, nonlin=torch.nn.ReLU,
                                                      hsizes=[64 for h in range(2)])
        self.dynamics = dynamics if dynamics != None else Dynamics(env, rb=self.rb, linear_dynamics=linear_dynamics)
        self.l = InputConcat(l) if l != None else InputConcat(PDQuadraticStageCost(self.ny, self.nu))
        # self.l = InputConcat(l) if l != None else InputConcat(blocks.MLP(self.ny + self.nu, 1, bias=True,
        #                                                   linear_map=torch.nn.Linear, nonlin=torch.nn.ReLU,
        #                                                   hsizes=[64 for h in range(2)]))
        # self.V = V if V != None else PDQuadraticTerminalCost(self.ny)                                        
        self.V = V if V != None else blocks.MLP(self.ny, 1, bias=True,
                                                          linear_map=torch.nn.Linear, nonlin=torch.nn.ReLU,
                                                          hsizes=[64 for h in range(2)])
        # self.x_bias = GoalBias(self.nx)
        # self.u_bias = GoalBias(self.nu)


        self.xlim = xlim # np.array(2,n) 0-lower, 1-upper
        self.ulim = ulim # np.array(2,m) 0-lower, 1-upper

        # self.x_bias_node = Node(self.x_bias, ['x'], ['x_bias'])
        # self.u_bias_node = Node(self.u_bias, ['u'], ['u_bias'])
        self.goal_node = Node(self.goal_map, ['x'], ['z'])
        self.mu_node = Node(self.mu, ['x'], ['u'], name='mu')
        self.dx_node = Node(self.dynamics.dx, ['x','u'],['x'])
        self.l_node = Node(self.l, ['z','u'],['l'])
        self.V_node = Node(self.V, ['z'],['V'])
        self.model = System([self.goal_node, self.mu_node, self.dx_node, self.l_node, self.V_node], nsteps=self.H + 2)
        self.model_kwargs = {'dtype' : list(self.model.parameters())[0].dtype,
                             'device' : list(self.model.parameters())[0].device,}

        # Formulate problem
        self.loss = loss
        self.scale = scale
        self.lpred = variable('l')
        self.Vpred = variable('V')
        self.l_loss = Objective(var=(self.H+1.)*self.lpred[:, :-1, :], name='stage_loss')
        self.V_loss = Objective(var=(self.Vpred[:, [-1], :]), name='terminal_loss')
        self.constraints = [] if ((self.xlim is None) and (self.ulim is None)) else self._constraints()
        self.problem = self._problem()

        # Setup optimizer
        if opt == "Adam":
            self.opt = optim.Adam(self.mu_node.parameters(), lr=lr)
        else:
            self.opt = optim.AdamW(self.mu_node.parameters(), lr=lr)

    def forward(self,x): 
        return self.mu(x)
    
    def train(self, trainer_kwargs=None, n_samples=10000, batch_size=256):
        train_loader = self._train_loader(n_samples, batch_size)
        trainer_kwargs = trainer_kwargs if trainer_kwargs != None else {'epochs':1, 'epoch_verbose':5, 'patience':1}
        trainer = Trainer(self.problem, train_loader,
                          optimizer=self.opt,
                          train_metric='train_loss',
                          eval_metric='train_loss',
                          **trainer_kwargs) # can add a test loss, but the dataset is constantly being updated anyway
        trainer.current_epoch = 2
        self.best_model = trainer.train() # output is a deepcopy
        return
    
    def _train_loader(self, n_samples, batch_size):

        # need to coordinate the number of epoch and batch size
        batch = self.rb.sample(n_samples)
        data = {}
        data['x'] = batch.observations.unsqueeze(1).to(**self.model_kwargs)
        datadict = DictDataset(data)

        train_loader = DataLoader(datadict, batch_size=batch_size, shuffle=True, collate_fn=datadict.collate_fn)
        return train_loader
    
    def _constraints(self):        
        x = variable('x')
        u = variable('u')
        constraints = []

        if self.xlim is not None:
            state_lower_bound_penalty = self.scale * (x > self.xlim[0])
            state_upper_bound_penalty = self.scale * (x < self.xlim[1])
            state_lower_bound_penalty.name = 'x_min'
            state_upper_bound_penalty.name = 'x_max'

            constraints.append(state_lower_bound_penalty)
            constraints.append(state_upper_bound_penalty)

        if self.ulim is not None:
            action_lower_bound_penalty = self.scale * (u > self.ulim[0])
            action_upper_bound_penalty = self.scale * (u < self.ulim[1])
            action_lower_bound_penalty.name = 'u_min'
            action_upper_bound_penalty.name = 'u_max'

            constraints.append(action_lower_bound_penalty)
            constraints.append(action_upper_bound_penalty)

        return constraints
    
    def _problem(self):
        if self.loss == 'penalty':
            self.obj = PenaltyLoss([self.l_loss, self.V_loss], self.constraints)
        elif self.loss == 'lagrange':
            """
            I don't think this was ever fully implemented by Neuromancer
                1) The lagrange multiplier logic only works if the batch_size is the size of the train_loader (n_samples)
                2) The logic for applying the lagrange multipliers in loss.py uses input_dict['input'],
                    but the key ['input'] is never added to input_dict nor referenced in source outside of loss.py
            """
            raise NotImplementedError
            assert n_samples == batch_size
            train_loader = self._train_loader(n_samples, batch_size)
            self.obj = AugmentedLagrangeLoss([self.l_loss, self.V_loss], self.constraints, train_data=train_loader)
        else: # unconstrained
            self.obj = PenaltyLoss([self.l_loss, self.V_loss], [])

        problem = Problem([self.model], self.obj)
        return problem

if __name__ == "__main__":
    import os
    import random
    import time
    from dataclasses import dataclass
    import tyro

    from stable_baselines3.common.buffers import ReplayBuffer

    from dynamics import Dynamics
    from mpcomponents import QuadraticStageCost, QuadraticTerminalCost, LinearDynamics, LinearPolicy, GoalMap
    from templates import LQREnv
    from utils import calc_K, calc_P, fill_rb

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
        device=kwargs['device'],
        handle_timeout_termination=False,
        n_envs=n_envs
    )

    """ System information """
    n, m = envs.get_attr("n")[0], envs.get_attr("m")[0]
    A_env, B_env = envs.get_attr("A")[0], envs.get_attr("B")[0]
    Q, R = envs.get_attr("Q")[0], envs.get_attr("R")[0]
    
    K_opt = calc_K(A_env, B_env, Q, R)

    """ Model setup """
    A = A_env
    B = B_env
    K = np.random.uniform(-1., 1., (m,n)).astype(np_kwargs['dtype'])
    P = calc_P(A, B, Q, R).astype(np_kwargs['dtype'])
    
    mpc_horizon = 1
    l = QuadraticStageCost(n, m, Q, R)
    V = QuadraticTerminalCost(n, P)
    f = LinearDynamics(n, m, A, B)
    mu = LinearPolicy(n, m, K)

    concat_f = InputConcat(f)
    dynamics = Dynamics(envs, rb, dx=concat_f)

    xlim = np.vstack([-np.ones(n), np.ones(n)])
    ulim = np.vstack([-np.ones(m), np.ones(m)])
    dpcontrol = DPControl(envs, rb=rb, H=mpc_horizon, dynamics=dynamics, l=l, V=V, mu=mu, goal_map=GoalMap(), xlim=xlim, ulim=ulim, loss='penalty', scale=10.)

    """ Learning ficticious controller """
    K_init = K.copy()
    print(f"Before training: 'Distance' from optimal gain: {np.linalg.norm(K_opt - K_init, 'fro')}")

    """ Scheme 1 """
    trainer_kwargs = {'epochs':400, 'epoch_verbose':10, 'patience':400,}
    obs, _ = envs.reset(seed=args.seed)
    obs = fill_rb(rb, envs, obs, n_samples=int(1e4))
    dpcontrol.train(trainer_kwargs=trainer_kwargs, n_samples=int(1e4), batch_size=args.batch_size)
    K_learn = mu.K.detach().numpy()
    print(f"Iter: 'Distance' from optimal gain: {np.linalg.norm(K_opt - K_learn, 'fro')}")

    """ Scheme 2 """
    # trainer_kwargs = {'epochs':1, 'epoch_verbose':1, 'patience':1,}
    # for i in range(1000):
    #     obs = fill_rb(rb, envs, obs, n_samples=args.batch_size)
    #     dpcontrol.train(trainer_kwargs=trainer_kwargs, n_samples=args.batch_size, batch_size=args.batch_size)

    #     if (i % 100) == 0:
    #         K_learn = mu.K.detach().numpy()
    #         print(f"Iter {i}: 'Distance' from optimal gain: {np.linalg.norm(K_opt - K_learn, 'fro')}")

    print(f"K' == K^*:\n{np.isclose(K_opt, K_learn)}")