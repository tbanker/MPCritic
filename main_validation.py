import os
import sys
from datetime import date
import random
import time
import numpy as np
import torch
from torch import optim

import gymnasium as gym
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from dataclasses import dataclass

from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, PDQuadraticTerminalCost, LinearDynamics, LinearPolicy, GoalMap
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

def validation_test(
    args,
    make_env,
    learn_dynamics, 
    learn_dpcontrol,
    learn_mpcritic,
    pd_V,
    seed,
    batch_size,
    n_batches,
    n_sysid_batches,
    n_samples,
    sampling,
    save_results,
    save_less=False,
):
    """ User settings: """

    exp_kwargs = {
        'seed':seed, 'batch_size':batch_size, 'n_batches':n_batches, 'n_sysid_batches':n_sysid_batches, 'n_samples':n_samples, 'sampling':sampling, 'pd_V':pd_V #, 'f_unc_scale':f_unc_scale
        }
    
    trainer_kwargs = {
        'epochs':1, 'epoch_verbose':1, 'patience':1,
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

    K_opt = calc_K(A_env, B_env, Q, R).astype(np_kwargs['dtype'])
    P_opt = calc_P(A_env, B_env, Q, R).astype(np_kwargs['dtype'])

    """ Model setup """
    A_mpc = np.random.normal(size=(n,n)).astype(np_kwargs['dtype']) if learn_dynamics else A_env.copy()
    B_mpc = np.random.normal(size=(n,m)).astype(np_kwargs['dtype']) if learn_dynamics else B_env.copy()
    Q_mpc, R_mpc = Q.copy(), R.copy()
    K = np.random.normal(size=(m,n)).astype(np_kwargs['dtype']) if learn_dpcontrol else K_opt
    if pd_V:
        L = np.random.normal(size=(n,n)).astype(np_kwargs['dtype'])
    else:
        P = np.random.uniform(size=(n,n)).astype(np_kwargs['dtype']) if learn_mpcritic else P_opt.copy()

    H = 1
    lr = 0.001
    V = PDQuadraticTerminalCost(n, L) if pd_V else QuadraticTerminalCost(n, P)
    l = QuadraticStageCost(n, m, Q_mpc, R_mpc)
    f = LinearDynamics(n, m, A_mpc, B_mpc)
    mu = LinearPolicy(n, m, K)

    concat_f = InputConcat(f)
    dynamics = Dynamics(envs, rb, dx=concat_f, lr=lr, opt="AdamW")

    dpcontrol = DPControl(envs, H=H, rb=rb, dynamics=dynamics, V=V, l=l, mu=mu, goal_map=GoalMap(), lr=lr, opt="AdamW")

    critic = MPCritic(dpcontrol)
    critic.setup_mpc()

    """ Learning Q-function """
    critic.requires_grad_(True)
    mse = torch.nn.MSELoss(reduction='mean')
    critic_optimizer = optim.AdamW(list(V.parameters()), lr=lr)
    p_true = np.concatenate([A_env, B_env], axis=1)

    results = {
        "exp_kwargs" : exp_kwargs,
        "A_env" : A_env,
        "B_env" : B_env,
        "Q" : Q,
        "R" : R,
        "P_opt" : P_opt,
        "K_opt" : K_opt,
        "A_lrn" : [],
        "B_lrn" : [],
        "K_lrn" : [],
        "P_lrn" : [],
    }

    results["A_lrn"].append(f.A.clone().detach().numpy())
    results["B_lrn"].append(f.B.clone().detach().numpy())
    results["K_lrn"].append(mu.K.clone().detach().numpy())
    results["P_lrn"].append(V.P.clone().detach().numpy())

    obs, _ = envs.reset(seed=args.seed)
    obs = fill_rb(rb, envs, obs, policy=None, sampling=sampling, n_samples=n_samples)

    for i in range(1,n_batches+1):
        
        batch = rb.sample(args.batch_size)

        with HiddenPrints():
            if learn_dynamics:
                critic.dpcontrol.dynamics.train(trainer_kwargs=trainer_kwargs, n_samples=batch_size, batch_size=batch_size)
            if i > n_sysid_batches:
                if learn_mpcritic:
                    with torch.no_grad():
                        q_targ = batch.rewards + critic(s=batch.next_observations)
                    q_pred = critic(s=batch.observations, a=batch.actions)
                    td_loss = mse(q_pred, q_targ)

                    critic_optimizer.zero_grad()
                    td_loss.backward()
                    critic_optimizer.step()
                if learn_dpcontrol:
                    critic.dpcontrol.train(trainer_kwargs=trainer_kwargs, n_samples=batch_size, batch_size=batch_size)                  

        if (i % 100) == 0:
            p_learn = np.concatenate([f.A.detach().numpy(), f.B.detach().numpy()], axis=1)
            print(f"Iter {i}: 'Distance' from...\n \
                optimal value function: {np.mean(np.abs(P_opt - V.P.detach().numpy()))}\n \
                optimal gain: {np.mean(np.abs(K_opt - mu.K.detach().numpy()))}\n \
                true model: {np.mean(np.abs(p_true - p_learn))}")
            if save_less:
                results["A_lrn"].append(f.A.clone().detach().numpy())
                results["B_lrn"].append(f.B.clone().detach().numpy())
                results["K_lrn"].append(mu.K.clone().detach().numpy())
                results["P_lrn"].append(V.P.clone().detach().numpy())

    print(f'P == P^*:\n{np.isclose(P_opt, V.P.detach().numpy())}')

    if save_results:
        results["A_lrn"] = np.array(results["A_lrn"])
        results["B_lrn"] = np.array(results["B_lrn"])
        results["K_lrn"] = np.array(results["K_lrn"])
        results["P_lrn"] = np.array(results["P_lrn"])

        file_name = f"seed={seed}.pt"
        save_dir = os.path.join(os.path.dirname(__file__), "runs", "validation", f"{date.today()}_n={n}_m={m}_f={learn_dynamics}_mu={learn_dpcontrol}_V={learn_mpcritic}_PD={pd_V}")
        file_path = os.path.join(save_dir, file_name)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(results, file_path)

if __name__ == '__main__':
    seeds = list(range(20))
    exp_dicts = {
        'learn_f_mu_V_pd' : {'learn_dynamics':True,  'learn_dpcontrol':True,  'learn_mpcritic':True,  'pd_V':True, },
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
                            'goal_map': goal_map, 'smooth_reward': False, 'sa_reward': True, 'sa_reward_scale': 1000,
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
                validation_test(
                    args,
                    make_env,
                    learn_dynamics = exp_dict['learn_dynamics'],
                    learn_dpcontrol = exp_dict['learn_dpcontrol'],
                    learn_mpcritic = exp_dict['learn_mpcritic'],
                    pd_V = exp_dict['pd_V'],

                    seed = seed,
                    batch_size = 256,
                    n_batches = 100000,
                    n_sysid_batches = 0,
                    n_samples = 100000,
                    sampling = "Uniform",

                    save_results=True,
                    save_less=True,
                )

