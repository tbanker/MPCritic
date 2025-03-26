import os
import sys
from datetime import date
import random
import time
import numpy as np
import torch
from torch import optim
from do_mpc.differentiator import DoMPCDifferentiator, NLPDifferentiator
import casadi as ca
from neuromancer.modules.blocks import MLP_bounds

import gymnasium as gym
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from dataclasses import dataclass

from modules.templates import LQREnv, template_LQR_model, template_conLQR_mpc
from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, LinearDynamics, LinearPolicy, GoalMap
from modules.mpcritic import MPCritic, InputConcat
from modules.dynamics import Dynamics
from modules.dpcontrol import DPControl
from modules.utils import calc_K, calc_P, fill_rb

np_kwargs = {'dtype' : np.float32}
kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

""" CleanRL setup """
gym.register(
    id="gymnasium_env/LQR-v0",
    entry_point=LQREnv,
    kwargs={'max_timesteps': 1} # designed to randomly initialize state, take action, and then restart environment
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

def scalability_test(
    H,
    lim,
    set_initial_guess,
    mu_class,
    n_hidden,
    hidden_nodes,
    seed,
    batch_size,
    n_batches,
    n_samples,
    sampling,
    save_results,
):
    """ User settings: """

    exp_kwargs = {
        'seed':seed, 'batch_size':batch_size, 'n_batches':n_batches, 'n_samples':n_samples, 'sampling':sampling,
        }
    
    trainer_kwargs = {
        'epochs':1, 'epoch_verbose':1, 'patience':1,
    }

    args = tyro.cli(Args)
    args.seed = seed
    args.buffer_size = n_samples
    args.batch_size = batch_size

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

    """ Model setup """
    A_mpc = A_env.copy()
    B_mpc = B_env.copy()
    Q_mpc, R_mpc = Q.copy(), R.copy()
    K = K_opt
    P = P_opt.copy()

    lr = 0.001
    xlim = np.vstack([-lim*np.ones(n), lim*np.ones(n)]) # np.vstack([-np.inf*np.ones(n), np.inf*np.ones(n)])
    ulim = np.vstack([-lim*np.ones(m), lim*np.ones(m)]) # np.vstack([-np.inf*np.ones(m), np.inf*np.ones(m)])

    critic_start = time.time()
    f = LinearDynamics(n, m, A_mpc, B_mpc)
    concat_f = InputConcat(f)
    dynamics = Dynamics(envs, rb, dx=concat_f, lr=lr)

    # dpcontrol = DPControl(envs, rb, mpc_horizon, dynamics, l, V, mu, lr=lr)
    mu_start = time.time()
    if mu_class == "MLP_bounds":
        mu = MLP_bounds(n, m, bias=True,
                        linear_map=torch.nn.Linear, nonlin=torch.nn.ReLU,
                        hsizes=[hidden_nodes for h in range(n_hidden)],
                        min=torch.tensor(ulim[0]),
                        max=torch.tensor(ulim[1]))
    mu_setup = time.time() - mu_start

    V = QuadraticTerminalCost(n, P)
    l = QuadraticStageCost(n, m, Q_mpc, R_mpc)
    dpcontrol = DPControl(envs, H=H, rb=rb, dynamics=dynamics, V=V, l=l, mu=mu, goal_map=GoalMap(), lr=lr, xlim=xlim, ulim=ulim, opt="AdamW")
    dpcontrol_setup = time.time() - critic_start
    critic = MPCritic(dpcontrol)
    critic_setup = time.time() - critic_start


    unc_p = {'A' : A_mpc,
             'B' : B_mpc,
             'P' : P_opt}
    model = template_LQR_model(n, m)

    """ Learning Q-function """
    critic.requires_grad_(True)
    critic_params = list(V.parameters())+list(f.parameters())
    critic_optimizer = optim.Adam(critic_params, lr=lr)

    results = {
        "exp_kwargs" : exp_kwargs,
        "A" : A_env,
        "B" : B_env,
        "Q" : Q,
        "R" : R,
        "mu_fwd" : [],
        "mu_bkwd" : [],
        "mu_setup": np.array([mu_setup]),
        "dpcontrol_setup": np.array([dpcontrol_setup]),
        "critic_setup": np.array([critic_setup]),
        "mpc_setup": [],
        "mpc_init" : [],
        "mpc_fwd" : [],
        "mpc_bkwd" : [],
        "avg_mpc_init" : [],
        "avg_mpc_fwd" : [],
        "avg_mpc_bkwd" : [],
    }

    obs, _ = envs.reset(seed=args.seed)
    obs = fill_rb(rb, envs, obs, policy=None, sampling=sampling, n_samples=n_samples)

    for i in range(n_batches):
        
        batch = rb.sample(args.batch_size)
        critic_optimizer.zero_grad()

        input = batch.observations
        start = time.time()
        output = critic.dpcontrol.mu(input) # critic(input)
        mu_fwd = time.time() - start
        if torch.any(output < torch.tensor(ulim[0])) or torch.any(output > torch.tensor(ulim[1])):
            raise ValueError("Outside of constraints; unfair comparison")

        loss = output.sum()
        start = time.time()
        loss.backward()
        mu_bkwd = time.time() - start
        print(f"mu forward: {mu_fwd}\nmu backward: {mu_bkwd}")

        # time MPC
        mpc_init, mpc_fwd, mpc_bkwd = 0, 0, 0

        start = time.time()
        mpc = template_conLQR_mpc(model, H=H, mpc_p=unc_p, xlim=xlim, ulim=ulim)
        mpc_diff = DoMPCDifferentiator(mpc)
        mpc_setup = time.time() - start

        for i in range(args.batch_size):

            x0 = batch.observations[[i]].mT.numpy()
            mpc.x0 = x0

            if set_initial_guess:
                start = time.time()
                mpc.set_initial_guess()
                mpc_init += time.time() - start

            with HiddenPrints():
                input = x0
                start = time.time()
                mpc.make_step(input)
                mpc_fwd += time.time() - start

                start = time.time()
                dx_dp_num, dlam_dp_num = mpc_diff.differentiate()
                mpc_bkwd += time.time() - start
        print(f"mpc forward: {mpc_fwd}\nmpc backward: {mpc_bkwd}")
        avg_mpc_init = mpc_init / args.batch_size
        avg_mpc_fwd = mpc_fwd / args.batch_size
        avg_mpc_bkwd = mpc_bkwd / args.batch_size

        results["mu_fwd"].append(mu_fwd)
        results["mu_bkwd"].append(mu_bkwd)
        results["mpc_setup"].append(mpc_setup)
        results["mpc_fwd"].append(mpc_fwd)
        results["mpc_bkwd"].append(mpc_bkwd)
        results["avg_mpc_init"].append(avg_mpc_init)
        results["avg_mpc_fwd"].append(avg_mpc_fwd)
        results["avg_mpc_bkwd"].append(avg_mpc_bkwd)

    if save_results:
        results["mu_fwd"] = np.array(results["mu_fwd"])
        results["mu_bkwd"] = np.array(results["mu_bkwd"])
        results["mpc_setup"] = np.array(results["mpc_setup"])
        results["mpc_init"] = np.array(results["mpc_init"])
        results["mpc_fwd"] = np.array(results["mpc_fwd"])
        results["mpc_bkwd"] = np.array(results["mpc_bkwd"])
        results["avg_mpc_init"] = np.array(results["avg_mpc_init"])
        results["avg_mpc_fwd"] = np.array(results["avg_mpc_fwd"])
        results["avg_mpc_bkwd"] = np.array(results["avg_mpc_bkwd"])

        file_name = f"seed={seed}.pt"
        save_dir = os.path.join(os.path.dirname(__file__), "runs", "scalability", f"{date.today()}_n={n}_m={m}_H={H}_lim={lim}_set_initial_guess={set_initial_guess}")
        file_path = os.path.join(save_dir, file_name)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(results, file_path)

if __name__ == '__main__':
    from modules.utils import stable, controllable
    from scipy.linalg import block_diag

    # redo seeds 0-2
    seeds = [11]+list(range(10))
    exp_dicts = {
        'H=1_set_initial_guess=True_MLP_bounds_2x100' : {'H':1, 'lim':10., 'set_initial_guess':True, 'mu_class':'MLP_bounds', 'n_hidden':2, 'hidden_nodes':100},
    }

    n_list = [2**i for i in range(2,8)]
    m_list = n_list
    for seed in seeds:
        for n, m in zip(n_list, m_list):
            A = np.diag(1.01*np.ones(n)) + np.diag(0.01*np.ones(n-1), k=1) + np.diag(0.01*np.ones(n-1), k=-1)
            if n > m:
                B = np.diag(np.ones(m))
                B = np.concatenate([B]*(n//m), axis=0)
            elif n < m:
                B = np.diag(np.ones(n))
                B = np.concatenate([B]*(m//n), axis=1)
            else:
                B = np.diag(np.ones(m))
            A, B = A.astype(np_kwargs['dtype']), B.astype(np_kwargs['dtype'])
            stable(A)
            controllable(A, B)

            gym.register(
                id="gymnasium_env/LQR-v0",
                entry_point=LQREnv,
                kwargs={'A' : A,
                        'B' : B,
                        'Q' : np.diag(np.ones(A.shape[0], **np_kwargs)),
                        'R' : 1000*np.diag(np.ones(B.shape[1], **np_kwargs)),
                        'max_timesteps': 1} # designed to randomly initialize state, take action, and then restart environment
            )
            
            for exp_dict in exp_dicts.values():
                scalability_test(
                    H = exp_dict['H'],
                    lim = exp_dict['lim'],
                    set_initial_guess = exp_dict['set_initial_guess'],
                    n_hidden = exp_dict['n_hidden'],
                    mu_class = exp_dict['mu_class'],
                    hidden_nodes = exp_dict['hidden_nodes'],

                    seed = seed,
                    batch_size = 256,
                    n_batches = 1,
                    n_samples = 256,
                    sampling = "Uniform",

                    save_results = True,
                )