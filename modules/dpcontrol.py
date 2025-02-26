import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# neuromancer stuff
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dynamics import integrators
from neuromancer.constraint import variable, Objective
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.dataset import DictDataset
from neuromancer.trainer import Trainer
from neuromancer.psl import signals
import torch.optim as optim
from torch.utils.data import DataLoader

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
    def __init__(self, env, rb, H, dynamics, l, V, mu=None):
        super().__init__()

        self.env = env
        self.rb = rb

        # Configure network
        self.nx = np.array(env.single_observation_space.shape).prod()
        self.nu = np.array(env.single_action_space.shape).prod()
        self.H = H # mpc horizon per do-mpc

        self.mu = mu if mu != None else blocks.ResMLP(self.nx, self.nu, bias=True,
                                                      linear_map=torch.nn.Linear, nonlin=torch.nn.SiLU,
                                                      hsizes=[64 for h in range(2)])
        self.dynamics = dynamics
        self.l = InputConcat(l)
        self.V = V

        self.mu_node = Node(self.mu, ['x'], ['u'], name='mu')
        self.dx_node = Node(self.dynamics.dx, ['x','u'],['x'])
        self.l_node = Node(self.l, ['x','u'],['l'])
        self.V_node = Node(self.V, ['x'],['V'])
        self.model = System([self.mu_node, self.dx_node, self.l_node, self.V_node], nsteps=self.H + 2)
        self.model_kwargs = {'dtype' : list(self.model.parameters())[0].dtype,
                             'device' : list(self.model.parameters())[0].device,}

        # Formulate problem
        self.lpred = variable('l')
        self.Vpred = variable('V')
        self.l_loss = Objective(var=(self.H+1.)*self.lpred[:, :-1, :], name='stage_loss')
        self.V_loss = Objective(var=(self.Vpred[:, [-1], :]), name='terminal_loss')
        self.obj = PenaltyLoss([self.l_loss, self.V_loss], [])
        self.problem = Problem([self.model], self.obj)

        # Setup optimizer
        self.opt = optim.Adam(self.mu_node.parameters(), 0.001)

    def forward(self,x):
        return self.mu(x)
    
    def train(self):
        train_loader = self._train_loader()
        trainer = Trainer(self.problem, train_loader,
                          optimizer=self.opt,
                          epochs=1000, epoch_verbose=100,
                          patience=400,
                          train_metric='train_loss', eval_metric='train_loss') # can add a test loss, but the dataset is constantly being updated anyway
        self.best_model = trainer.train() # output is a deepcopy
        return
    
    def _train_loader(self):

        # need to coordinate the number of epoch and batch size
        batch = self.rb.sample(1000)
        data = {}
        data['x'] = batch.observations.unsqueeze(1).to(**self.model_kwargs)
        # data = {**data, **self.mu_node(data)}
        datadict = DictDataset(data)

        train_loader = DataLoader(datadict, batch_size=64, shuffle=True, collate_fn=datadict.collate_fn)
        return train_loader

if __name__ == "__main__":
    import os
    import random
    import time
    from dataclasses import dataclass
    import tyro

    import numpy as np
    import gymnasium as gym
    from stable_baselines3.common.buffers import ReplayBuffer

    from mpcomponents import QuadraticStageCost, QuadraticTerminalCost, LinearDynamics, LinearPolicy
    from dynamics import Dynamics
    from templates import template_linear_model, LQREnv
    from utils import calc_K, calc_P

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
        buffer_size: int = int(1e3)
        """the replay memory buffer size"""
        gamma: float = 0.99
        """the discount factor gamma"""
        tau: float = 0.005
        """target smoothing coefficient (default: 0.005)"""
        batch_size: int = 256
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

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device=kwargs['device'],
        handle_timeout_termination=False,
    )

    """ User settings: """
    learn_dynamics = False

    """ System information """
    b = 1
    n = 2
    m = n
    A_env, B_env = np.diag(np.ones(n, **np_kwargs)), np.diag(np.ones(m, **np_kwargs)),
    
    Q, R = np.diag(np.ones(n, **np_kwargs)), np.diag(np.ones(m, **np_kwargs))
    A = np.random.uniform(-1., 1., (n,n)).astype(np_kwargs['dtype']) if learn_dynamics else A_env
    B = np.random.uniform(-1., 1., (n,m)).astype(np_kwargs['dtype']) if learn_dynamics else B_env
    K = np.random.uniform(-1., 1., (m,n)).astype(np_kwargs['dtype'])
    P = calc_P(A, B, Q, R).astype(np_kwargs['dtype'])
    
    """ DPControl information """
    mpc_horizon = 1
    l = QuadraticStageCost(n, m, Q, R)
    V = QuadraticTerminalCost(n, P)
    f = LinearDynamics(n, m, A, B)
    mu = LinearPolicy(n, m, K)

    concat_f = InputConcat(f)
    dynamics = Dynamics(envs, rb, dx=concat_f)

    dpcontrol = DPControl(envs, rb, mpc_horizon, dynamics, l, V, mu)

    """ Fill replay buffer """
    obs, _ = envs.reset(seed=args.seed)
    for _ in range(args.buffer_size):
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

    """ Learning environment dynamics """
    if learn_dynamics:
        p_true = np.concat([A_env, B_env], axis=1)
        p_init = np.concat([f.A.detach().numpy(), f.B.detach().numpy()], axis=1)
        print(f"Before training: 'Distance' from true dynamics: {np.linalg.norm(p_true - p_init, 'fro')}")
        dynamics.train()
        p_learn = np.concat([f.A.detach().numpy(), f.B.detach().numpy()], axis=1)
        print(f"After training: 'Distance' from true dynamics: {np.linalg.norm(p_true - p_learn, 'fro')}")

    """ Learning ficticious controller """
    K_opt = calc_K(A_env, B_env, Q, R)
    K_init = K
    print(f"Before training: 'Distance' from optimal gain: {np.linalg.norm(K_opt - K_init, 'fro')}")
    dpcontrol.train()
    K_learn = mu.K.detach().numpy()
    print(f"After training: 'Distance' from true dynamics: {np.linalg.norm(K_opt - K_learn, 'fro')}")
    print(f"K' == K^*:\n{np.isclose(K_opt, K_learn)}")