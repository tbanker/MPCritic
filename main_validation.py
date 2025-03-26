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

from modules.templates import LQREnv
from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, PDQuadraticTerminalCost, LinearDynamics, LinearPolicy, GoalMap
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

def validation_test(
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

    unc_p = {'A' : [A_mpc],
             'B' : [B_mpc]} # 1 uncertainty scenario considered
    critic = MPCritic(dpcontrol, unc_p=unc_p)
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
    from modules.utils import stable, controllable
    from scipy.linalg import block_diag

    seeds = list(range(6,100))
    exp_dicts = {
        # 'learn_f' :         {'learn_dynamics':True,  'learn_dpcontrol':False, 'learn_mpcritic':False, 'pd_V':False, },
        # 'learn_mu' :        {'learn_dynamics':False, 'learn_dpcontrol':True,  'learn_mpcritic':False, 'pd_V':False, },
        # 'learn_V' :         {'learn_dynamics':False, 'learn_dpcontrol':False, 'learn_mpcritic':True,  'pd_V':False, },
        # 'learn_V_pd' :      {'learn_dynamics':False, 'learn_dpcontrol':False, 'learn_mpcritic':True,  'pd_V':True, },
        # 'learn_f_mu_V' :    {'learn_dynamics':True,  'learn_dpcontrol':True,  'learn_mpcritic':True,  'pd_V':False, },
        'learn_f_mu_V_pd' : {'learn_dynamics':True,  'learn_dpcontrol':True,  'learn_mpcritic':True,  'pd_V':True, },
    }

    n_list = [2**i for i in range(2,8)] # [4**i for i in range(1,4)] + [4]*2 + [4**i for i in range(2,4)]
    m_list = n_list # 3*[4] + [4**i for i in range(2,4)] + [4**i for i in range(2,4)]
    # n_list = [2**i for i in range(1,8,2)] + [2]*3 + [2**i for i in range(3,8,2)]
    # m_list = [2]*4 + [2**i for i in range(3,8,2)] + [2**i for i in range(3,8,2)]
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
                validation_test(
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

                    save_results = True,
                    save_less=True,
                )

