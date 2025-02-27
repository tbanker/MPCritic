import os
import sys
import random
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from copy import copy

import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer
from dataclasses import dataclass
import tyro

from modules.templates import template_linear_model, template_linear_simulator, LQREnv
from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, PDQuadraticTerminalCost, LinearDynamics, LinearPolicy
from modules.mpcritic import MPCritic, InputConcat
from modules.dynamics import Dynamics
from modules.dpcontrol import DPControl
from modules.utils import calc_K, calc_P, fill_rb

from neuromancer.dataset import DictDataset
from neuromancer.trainer import Trainer

np_kwargs = {'dtype' : np.float32}
kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}

""" User settings: """
learn_dynamics = False
learn_dpcontrol = False

""" CleanRL setup """
gym.register(
    id="gymnasium_env/LQR-v0",
    entry_point=LQREnv,
    # max_episode_steps=10 # LQR environment doesn't handle this correctly
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

K_opt = calc_K(A_env, B_env, Q, R).astype(np_kwargs['dtype'])
P_opt = calc_P(A_env, B_env, Q, R).astype(np_kwargs['dtype'])

""" MPCritic stuff """
# numpy arrays share memory with corresponding pytorch model params
mpc_horizon = 1
A_mpc = np.random.uniform(-1., 1., (n,n)).astype(np_kwargs['dtype']) if learn_dynamics else A_env.copy()
B_mpc = np.random.uniform(-1., 1., (n,m)).astype(np_kwargs['dtype']) if learn_dynamics else B_env.copy()
Q_mpc, R_mpc = Q.copy(), R.copy()

if learn_dpcontrol:
    K = np.random.uniform(-1., 1., (m,n)).astype(np_kwargs['dtype'])
    P = P_opt
    V = QuadraticTerminalCost(n, P)
else:
    K = K_opt
    L = np.random.uniform(-1., 1., (n,n)).astype(np_kwargs['dtype'])
    V = PDQuadraticTerminalCost(n, L)
l = QuadraticStageCost(n, m, Q_mpc, R_mpc)
f = LinearDynamics(n, m, A_mpc, B_mpc)
mu = LinearPolicy(n, m, K)

concat_f = InputConcat(f)
dynamics = Dynamics(envs, rb, dx=concat_f)

dpcontrol = DPControl(envs, rb, mpc_horizon, dynamics, l, V, mu)

template_model = template_linear_model(n, m)
unc_p = {'A' : [A_mpc],
         'B' : [B_mpc]} # 1 uncertainty scenario considered
model = template_linear_model(n, m)
critic = MPCritic(template_model, dpcontrol, unc_p)
critic.setup_mpc()
critic.requires_grad_(True)

""" Fill replay buffer """
obs, _ = envs.reset(seed=args.seed)
# for _ in range(1000):
#     actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
#     next_obs, rewards, terminations, truncations, infos = envs.step(actions)

#     real_next_obs = next_obs.copy()
#     for idx, trunc in enumerate(truncations):
#         if trunc:
#             real_next_obs[idx] = infos["final_observation"][idx]
#     rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

#     obs = next_obs

if learn_dynamics:
    """ Learning environment dynamics """
    p_true = np.concat([A_env, B_env], axis=1)
    p_init = np.concat([f.A.detach().numpy(), f.B.detach().numpy()], axis=1)
    print(f"Before training: 'Distance' from true dynamics: {np.linalg.norm(p_true - p_init, 'fro')}")
    critic.dynamics.train()
    p_learn = np.concat([f.A.detach().numpy(), f.B.detach().numpy()], axis=1)
    print(f"After training: 'Distance' from true dynamics: {np.linalg.norm(p_true - p_learn, 'fro')}")

if learn_dpcontrol:
    """ Learning ficticious controller """
    K_opt = calc_K(A_env, B_env, Q, R)
    K_init = K
    print(f"Before training: 'Distance' from optimal gain: {np.linalg.norm(K_opt - K_init, 'fro')}")
    dpcontrol.train()
    K_learn = mu.K.detach().numpy()
    print(f"After training: 'Distance' from optimal gain: {np.linalg.norm(K_opt - K_learn, 'fro')}")
    print(f"K' == K^*:\n{np.isclose(K_opt, K_learn)}")

else:
    """ Learning Q-function """
    mse = torch.nn.MSELoss(reduction='mean')
    critic_optimizer = optim.Adam(list(V.parameters()), lr=0.001)

    # Training loop
    for i in range(1000):

        """
        Currently, the following is not always true:
        f(torch.concat((batch.observations, batch.actions), dim=-1)) == batch.next_observations
        originally thought this was due to early truncation, but I think it may be due to termination as well?
        """
        obs = fill_rb(rb, envs, args, obs, n_transitions=args.batch_size)
        # obs = 3.*torch.randn((n_env, n), **kwargs)
        # with torch.no_grad():
        #     actions = mu(obs) # actions of optimal policy
        #     rewards = -l(torch.concat((obs,actions), dim=-1)).squeeze(1) # negative quadratic
        #     real_next_obs = f(torch.concat((obs,actions), dim=-1)) # nominal dynamics
        #     terminations = [None] * 64
        # rb.add(obs, real_next_obs, actions, rewards, terminations, {})
        
        batch = rb.sample(args.batch_size)
        
        q_targ = batch.rewards + critic(batch.next_observations)
        q_pred = critic(batch.observations, batch.actions)
        td_loss = mse(q_pred, q_targ)

        critic_optimizer.zero_grad()
        td_loss.backward()
        critic_optimizer.step()

        if (i % 100) == 0:
            print(f"Iter {i}: 'Distance' from optimal value function: {np.linalg.norm(P_opt - V.P.detach().numpy(), 'fro')}")

    print(f'P == P^*:\n{np.isclose(P_opt, V.P.detach().numpy())}')    