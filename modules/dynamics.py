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
from neuromancer.constraint import variable
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

class Dynamics(nn.Module):
    def __init__(self, env, rb, dx=None):
        super().__init__()

        self.env = env
        self.rb = rb

        # Configure network
        self.nx = np.array(env.single_observation_space.shape).prod()
        self.nu = np.array(env.single_action_space.shape).prod()
        self.dx = dx if dx != None else blocks.ResMLP(self.nx + self.nu, self.nx, bias=True,
                                                      linear_map=torch.nn.Linear, nonlin=torch.nn.SiLU,
                                                      hsizes=[64 for h in range(2)])
        self.system_node = Node(self.dx, ['x','u'],['xnext'])
        self.x_shift = Node(lambda x: x, ['xnext'], ['x'])
        self.model = System([self.system_node], nstep_key='u') # or nsteps=1
        self.model_eval = System([self.system_node, self.x_shift], nstep_key='u')
        self.model_kwargs = {'dtype' : list(self.model.parameters())[0].dtype,
                             'device' : list(self.model.parameters())[0].device,}

        # Formulate problem
        self.xpred = variable('xnext')
        self.xtrue = variable('xtrue')
        self.loss = (self.xpred == self.xtrue)^2 # confusingly, this refers to a constraint, not a Boolean
        self.obj = PenaltyLoss([self.loss], [])
        self.problem = Problem([self.model], self.obj)

        # Setup optimizer
        self.opt = optim.Adam(self.model.parameters(), 0.001)

    def forward(self,x,u):
        return self.dx(x,u)
    
    def train(self, trainer_kwargs, n_samples=1000, batch_size=256):
        train_loader = self._train_loader(n_samples, batch_size)
        trainer = Trainer(self.problem, train_loader,
                          optimizer=self.opt,
                          train_metric='train_loss',
                          eval_metric='train_loss',
                          **trainer_kwargs) # can add a test loss, but the dataset is constantly being updated anyway
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
    
    def rollout_eval(self):

        ## Add current run logger for wanbd

        obs, info = self.env.reset()
        signal = signals.prbs(10, self.nu, min=self.env.action_space.low, max=self.env.action_space.high, p=.9, rng=np.random.default_rng())/2
        trajectory = self.model_eval({'x': torch.Tensor(obs).unsqueeze(1).to(**self.model_kwargs),
                                      'u':torch.Tensor(signal[None,:]).to(**self.model_kwargs)})   
        
        mae = 0.0
        for u,x in zip(signal,trajectory['x'].squeeze().detach().numpy()):
            error = obs.squeeze() - x
            mae += np.abs(np.mean(error))
            obs, rewards, terminations, truncations, infos = envs.step([u])
            # print(s)

        print(mae/len(signal))
        return

if __name__ == "__main__":
    import os
    import random
    import time
    from dataclasses import dataclass
    import tyro

    from stable_baselines3.common.buffers import ReplayBuffer

    from mpcritic import InputConcat
    from mpcomponents import LinearDynamics
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
    A_env, B_env = envs.get_attr("A")[0], envs.get_attr("B")[0]
    Q, R = envs.get_attr("Q")[0], envs.get_attr("R")[0]

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

    """ Rollout eval """
    if n_envs == 1:
        dynamics.rollout_eval()

    """ Training """
    p_true = np.concatenate([A_env, B_env], axis=1)
    p_init = np.concatenate([f.A.detach().numpy(), f.B.detach().numpy()], axis=1)
    trainer_kwargs = {'epochs':1, 'epoch_verbose':1, 'patience':1,}

    obs, _ = envs.reset(seed=args.seed)
    for i in range(10000):
        obs = fill_rb(rb, envs, obs, n_samples=args.batch_size)
        dynamics.train(trainer_kwargs=trainer_kwargs, n_samples=args.batch_size, batch_size=args.batch_size)
 
        if (i % 100) == 0:
            p_learn = np.concatenate([f.A.detach().numpy(), f.B.detach().numpy()], axis=1)
            print(f"Iter {i}: 'Distance' from true model: {np.linalg.norm(p_true - p_learn, 'fro')}")

    print(f"A' == A_env:\n{np.isclose(A_env, f.A.detach().numpy())};\nB' == B_env:\n{np.isclose(B_env, f.B.detach().numpy())}")