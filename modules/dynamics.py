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
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.dataset import DictDataset
from neuromancer.trainer import Trainer
from neuromancer.psl import signals
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append('')
from modules.mpcomponents import LinearDynamics
from modules.utils import HiddenPrints


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

class Dynamics(nn.Module):
    def __init__(self, env, rb=None, dx=None, linear_dynamics=False, opt="AdamW", lr=0.001):
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
        if dx != None:
            self.dx = dx
        elif linear_dynamics:
            self.dx = InputConcat(LinearDynamics(self.nx, self.nu))
        else:
            self.dx = InputConcat(blocks.ResMLP(self.nx + self.nu, self.nx, bias=True,
                                                        linear_map=torch.nn.Linear, nonlin=torch.nn.ReLU,
                                                        hsizes=[64 for h in range(2)]))
        self.system_node = Node(self.dx, ['x','u'],['xnext'])
        self.x_shift = Node(lambda x: x, ['xnext'], ['x'])
        self.model = System([self.system_node], nstep_key='u') # or nsteps=1
        self.model_eval = System([self.system_node, self.x_shift], nstep_key='u')
        self.model_kwargs = {'dtype' : list(self.model.parameters())[0].dtype,
                             'device' : list(self.model.parameters())[0].device,}

        # Formulate problem
        self.xpred = variable('xnext')
        self.xtrue = variable('xtrue')
        self.loss = (self.xpred == self.xtrue)^2
        self.obj = PenaltyLoss([self.loss], [])
        self.problem = Problem([self.model], self.obj)

        # Setup optimizer
        if opt == "Adam":
            self.opt = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.opt = optim.AdamW(self.model.parameters(), lr=lr)

    def forward(self,x,u):
        return self.dx(x,u)
    
    def train(self, trainer_kwargs=None, n_samples=10000, batch_size=256, epochs=1, epoch_verbose=5, patience=1):
        train_loader = self._train_loader(n_samples, batch_size)
        trainer_kwargs = trainer_kwargs if trainer_kwargs != None else {'epochs':epochs, 'epoch_verbose':epoch_verbose, 'patience':patience}
        trainer = Trainer(self.problem, train_loader,
                          optimizer=self.opt,
                          train_metric='train_loss',
                          eval_metric='train_loss',
                          **trainer_kwargs)
        trainer.current_epoch = 2
        self.best_model = trainer.train() # output is a deepcopy
        return
    
    def _train_loader(self, n_samples, batch_size):

        batch = self.rb.sample(n_samples)
        data = {}
        data['x'] = batch.observations.unsqueeze(1).to(**self.model_kwargs)
        data['u'] = batch.actions.unsqueeze(1).to(**self.model_kwargs)
        data['xtrue'] = batch.next_observations.unsqueeze(1).to(**self.model_kwargs)
        datadict = DictDataset(data)

        train_loader = DataLoader(datadict, batch_size=batch_size, shuffle=True, collate_fn=datadict.collate_fn)
        return train_loader

if __name__ == "__main__":
    import os
    import random
    import time
    from dataclasses import dataclass
    import tyro

    from stable_baselines3.common.buffers import ReplayBuffer

    from modules.mpcritic import InputConcat
    from modules.mpcomponents import GoalMap
    from utils import fill_rb

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
        buffer_size: int = int(1e6)
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
        kwargs['device'],
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    """ System information """
    obs, _ = envs.reset(seed=args.seed)
    b, n, m = envs.num_envs, envs.get_attr("n")[0], envs.get_attr("m")[0]
    A_env, B_env = envs.envs[0].simulator.p_fun(0)['A'].full().astype(np_kwargs['dtype']), envs.envs[0].simulator.p_fun(0)['B'].full().astype(np_kwargs['dtype'])
    Q, R = np.diag(np.ones(n)).astype(np_kwargs['dtype']), envs.envs[0].sa_reward_scale*np.diag(np.ones(m)).astype(np_kwargs['dtype'])

    """ Model setup """
    A = np.random.uniform(-1., 1., (n,n)).astype(np_kwargs['dtype'])
    B = np.random.uniform(-1., 1., (n,m)).astype(np_kwargs['dtype'])
    
    f = LinearDynamics(n, m, A, B)
    concat_f = InputConcat(f)
    dynamics = Dynamics(envs, rb, dx=concat_f)

    """ Model predictions """
    obs, _ = envs.reset(seed=args.seed)

    action = np.random.uniform(-1, 1, (b,m)).astype(np_kwargs['dtype'])# envs.action_space.sample()
    pred_dx = dynamics.dx(torch.from_numpy(obs).to(**kwargs), torch.from_numpy(action).to(**kwargs))
    pred_dyn = dynamics(torch.from_numpy(obs).to(**kwargs), torch.from_numpy(action).to(**kwargs))
    print(f"s'_dx == s'_dyn: {torch.allclose(pred_dx, pred_dyn)}")

    """ Training """
    p_true = np.concatenate([A_env, B_env], axis=1)
    p_init = np.concatenate([f.A.detach().numpy(), f.B.detach().numpy()], axis=1)
    trainer_kwargs = {'epochs':1, 'epoch_verbose':1, 'patience':1,}

    obs, _ = envs.reset(seed=args.seed)
    for i in range(10000):
        obs = fill_rb(rb, envs, obs, n_samples=args.batch_size)
        with HiddenPrints():
            dynamics.train(trainer_kwargs=trainer_kwargs, n_samples=args.batch_size, batch_size=args.batch_size)
 
        if (i % 100) == 0:
            p_learn = np.concatenate([f.A.detach().numpy(), f.B.detach().numpy()], axis=1)
            print(f"Iter {i}: 'Distance' from true model: {np.linalg.norm(p_true - p_learn, 'fro')}")

    print(f"A' == A_env:\n{np.isclose(A_env, f.A.detach().numpy())};\nB' == B_env:\n{np.isclose(B_env, f.B.detach().numpy())}")