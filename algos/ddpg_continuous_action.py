# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

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

# MPCritic stuff
import sys
sys.path.append('')
from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, PDQuadraticTerminalCost, LinearDynamics, LinearPolicy, GoalMap
from modules.mpcritic import MPCritic, InputConcat
from modules.dynamics import Dynamics
from modules.dpcontrol import DPControl
from modules.utils import calc_K, calc_P

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
    env_id: str = "lqr-v0" # revisit mujoco install
    """the environment id of the Atari game"""
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
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
    learning_starts: int = 5e2
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    # LQR specific arguments
    n = 4
    """state AND action dimension"""

    # MPCritic specific arguments
    critic_mode = "mpcritic"
    """choose between `mpcritic` or `vanilla`"""
    mpc_actor = True
    """use mpc actor based on mpcritic, or use a separately trained nn actor"""
    pretrain = True
    """TODO: spend more time optimizing f and mu after initial data collection but before RL training"""


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


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias



if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if "cstr" in args.env_id:
        goal_map = GoalMap(idx=[1], goal=0.6)
    else:
        goal_map = GoalMap()
    exp_path = f"runs/{run_name}/"
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name, exp_path, goal_map)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    actor_optimizer = optim.AdamW(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    if args.critic_mode == "vanilla":
        qf1 = QNetwork(envs).to(device)
        qf1_target = QNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        q_optimizer = optim.AdamW(list(qf1.parameters()), lr=args.learning_rate)
    elif args.critic_mode == "mpcritic":
        n, m = 4, 4
        A_env = (np.diag(1.01*np.ones(n)) + np.diag(0.01*np.ones(n-1), k=1) + np.diag(0.01*np.ones(n-1), k=-1)).astype(np.float32)
        B_env = np.diag(np.ones(m)).astype(np.float32)
        Q = np.diag(np.ones(n)).astype(np.float32)
        R = np.diag(np.ones(n)).astype(np.float32)

        K_opt = calc_K(A_env, B_env, Q, R).astype(np.float32)
        P_opt = calc_P(A_env, B_env, Q, R).astype(np.float32)

        l = QuadraticStageCost(n, m, Q, R)
        V = QuadraticTerminalCost(n, P_opt)
        f = LinearDynamics(n, m, A_env, B_env)
        concat_f = InputConcat(f)
        dynamics = Dynamics(envs, rb, dx=concat_f, lr=args.learning_rate)        
        mu = LinearPolicy(n, m, K_opt)
        dpcontrol = DPControl(envs, rb=rb, mu=mu, dynamics=dynamics, l=l, V=V, goal_map=goal_map, lr=args.learning_rate, xlim=np.array([envs.observation_space.low,envs.observation_space.high]), ulim=np.concatenate([envs.action_space.low,envs.action_space.high], axis=0)).to(device)
        dpcontrol_target = DPControl(envs, rb=rb, mu=mu, dynamics=dynamics, l=l, V=V, goal_map=goal_map, lr=args.learning_rate, xlim=np.array([envs.observation_space.low,envs.observation_space.high]), ulim=np.concatenate([envs.action_space.low,envs.action_space.high], axis=0)).to(device)
        # mu = blocks.MLP_bounds(insize=np.array(envs.single_observation_space.shape).prod(), outsize=np.array(envs.single_action_space.shape).prod(), bias=True, linear_map=torch.nn.Linear, nonlin=torch.nn.ReLU, hsizes=[100 for h in range(2)], min=torch.from_numpy(envs.action_space.low), max=torch.from_numpy(envs.action_space.high))
        # V = PDQuadraticTerminalCost(np.array(envs.single_observation_space.shape).prod())
        # dpcontrol = DPControl(envs, mu=mu, linear_dynamics=True, V=V, rb=rb, goal_map=goal_map, lr=args.learning_rate, xlim=np.array([envs.observation_space.low,envs.observation_space.high]), ulim=np.concatenate([envs.action_space.low,envs.action_space.high], axis=0)).to(device)
        # dpcontrol_target = DPControl(envs, mu=mu, linear_dynamics=True, V=V, rb=rb, goal_map=goal_map, lr=args.learning_rate, xlim=np.array([envs.observation_space.low,envs.observation_space.high]), ulim=np.concatenate([envs.action_space.low,envs.action_space.high], axis=0)).to(device)
        dpcontrol_target.load_state_dict(dpcontrol.state_dict())
        
        qf1 = MPCritic(dpcontrol).to(device)
        qf1.setup_mpc()
        qf1.requires_grad_(True) # just do it
        qf1_target = MPCritic(dpcontrol_target).to(device)
        qf1_params = list()
        for p in qf1.critic_parameters.values():
            qf1_params += list(p.parameters())
        q_optimizer = optim.AdamW(qf1_params, lr=args.learning_rate) 

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            if args.mpc_actor and args.critic_mode=="mpcritic":
                with torch.no_grad():
                    actions = qf1._rl_action(qf1.forward_mpc(qf1._mpc_state(obs)))
            else:
                with torch.no_grad():
                    actions = actor(torch.Tensor(obs).to(device))

            actions += torch.normal(0, actor.action_scale * args.exploration_noise)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if args.critic_mode == "mpcritic":
                if global_step % 10 == 0:
                    qf1.requires_grad_(True)
                    qf1.train_f_mu()
                if args.mpc_actor and (terminations or truncations):
                        qf1.online_mpc_update(qf1._mpc_state(obs), full=True)
                        qf1.requires_grad_(True)
                # qf1.online_mpc_update(qf1._mpc_state(obs), full=True)
                # qf1.requires_grad_(True)

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ddpg_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=args.exploration_noise,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DDPG", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
