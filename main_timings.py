import os
import sys
import random
from datetime import date
import time
import numpy as np
import torch
from torch import optim
from do_mpc.differentiator import DoMPCDifferentiator

import gymnasium as gym
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from dataclasses import dataclass
from neuromancer.modules.blocks import MLP_bounds

from envs.LQR.template_model import template_symbolic_model
from envs.LQR.template_mpc import template_mpc
from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, LinearDynamics, LinearPolicy, GoalMap
from modules.mpcritic import MPCritic, InputConcat
from modules.dynamics import Dynamics
from modules.dpcontrol import DPControl
from modules.utils import calc_K, calc_P, fill_rb, HiddenPrints

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

def scalability_test(
    args,
    make_env,
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
    Q_mpc, R_mpc = Q.copy(), R.copy()
    P = P_opt.copy()

    lr = 0.001
    xlim = np.vstack([-lim*np.ones(n), lim*np.ones(n)])
    ulim = np.vstack([-lim*np.ones(m), lim*np.ones(m)])

    critic_start = time.time()
    f = LinearDynamics(n, m, A_env, B_env)
    concat_f = InputConcat(f)
    dynamics = Dynamics(envs, rb=rb, dx=concat_f)

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

    obs = fill_rb(rb, envs, obs, policy=None, sampling=sampling, n_samples=n_samples)

    for i in range(n_batches):
        
        batch = rb.sample(args.batch_size)
        critic_optimizer.zero_grad()

        input = batch.observations
        start = time.time()
        output = critic.dpcontrol.mu(input) # critic(input)
        mu_fwd = time.time() - start

        loss = output.sum()
        start = time.time()
        loss.backward()
        mu_bkwd = time.time() - start
        print(f"mu forward: {mu_fwd}\nmu backward: {mu_bkwd}")

        # time MPC
        mpc_init, mpc_fwd, mpc_bkwd = 0, 0, 0

        start = time.time()
        model = template_symbolic_model(n, m)
        mpc = template_mpc(model, n_horizon=H)
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
    seeds = [11]+list(range(10))
    exp_dicts = {
        'H=1_set_initial_guess=True_MLP_bounds_2x100' : {'H':1, 'lim':1., 'set_initial_guess':True, 'mu_class':'MLP_bounds', 'n_hidden':2, 'hidden_nodes':100},
    }

    n_list = [2**i for i in range(2,8)]
    m_list = n_list
    for seed in seeds:
        for n, m in zip(n_list, m_list):     

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

                    model = template_model(n=n, m=n)
                    max_x = np.ones(n).flatten()
                    min_x = -np.ones(n).flatten() # writing like this to emphasize do-mpc sizing convention
                    max_u = np.ones(n).flatten()
                    min_u = -np.ones(n).flatten()
                    bounds = {'x_low' : min_x, 'x_high' : max_x, 'u_low' : min_u, 'u_high' : max_u}
                    goal_map = goal_map
                    num_steps = 1
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
            args.seed = seed
            args.n = n
              
            for exp_dict in exp_dicts.values():
                scalability_test(
                    args,
                    make_env,
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