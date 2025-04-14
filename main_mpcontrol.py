import os
import sys
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

from dataclasses import dataclass
import tyro
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer

from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, LinearDynamics, LinearPolicy, GoalMap
from modules.mpcritic import MPCritic, InputConcat
from modules.dynamics import Dynamics
from modules.dpcontrol import DPControl
from modules.utils import calc_K, calc_P

np_kwargs = {'dtype' : np.float32}
kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}

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

""" User settings: """
show_animation = True
store_results = False

""" Set initial state """
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
x0, _ = envs.reset(seed=args.seed)

""" Get configured do-mpc modules: """
b, n, m = envs.num_envs, envs.get_attr("n")[0], envs.get_attr("m")[0]
A_env, B_env = envs.envs[0].simulator.p_fun(0)['A'].full().astype(np_kwargs['dtype']), envs.envs[0].simulator.p_fun(0)['B'].full().astype(np_kwargs['dtype'])
Q, R = np.diag(np.ones(n)).astype(np_kwargs['dtype']), envs.envs[0].sa_reward_scale*np.diag(np.ones(m)).astype(np_kwargs['dtype'])

P_opt = calc_P(A_env, B_env, Q, R).astype(np_kwargs['dtype'])
K = calc_K(A_env, B_env, Q, R)

mpc_horizon = 10
l = QuadraticStageCost(n, m, Q, R)
V = QuadraticTerminalCost(n, Q)
f = LinearDynamics(n, m, A_env, B_env)
mu = LinearPolicy(n, m, K)

concat_f = InputConcat(f)
dynamics = Dynamics(envs, rb=rb, dx=concat_f)
xlim = np.vstack([-3.*np.ones(n), 3.*np.ones(n)])
ulim = np.vstack([-np.ones(m), np.ones(m)])
dpcontrol = DPControl(envs, H=10, rb=rb, dynamics=dynamics, l=l, V=V, mu=mu, goal_map=GoalMap(), xlim=xlim, ulim=ulim)

critic = MPCritic(dpcontrol)
critic.setup_mpc()

""" Setup graphic: """
fig, ax, graphics = do_mpc.graphics.default_plot(critic.mpc.data)
plt.ion()

""" Run MPC main loop: """
for k in range(50):
    u0 = critic._rl_action(critic.forward_mpc(critic._mpc_state(x0)))
    u0 = u0.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
    x0, _, _, _, _ = envs.step(u0)

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

input('Press any key to exit.')