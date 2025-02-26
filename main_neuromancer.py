import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from copy import copy

from modules.templates import template_linear_model, template_linear_simulator, LQREnv
from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, PDQuadraticTerminalCost, LinearDynamics, LinearPolicy
from modules.mpcritic import MPCritic, InputConcat
from modules.dynamics import Dynamics
from modules.utils import calc_K, calc_P

from neuromancer.dataset import DictDataset
from neuromancer.trainer import Trainer

import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer

np_kwargs = {'dtype' : np.float32}
kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}
seed = 43
np.random.seed(seed)
torch.manual_seed(seed)

""" User settings: """
learn_dynamics = True
mode = 'critic' # 'dpc' or 'critic'

""" Get configured do-mpc modules: """
num_envs, n, m = 1, 4, 2
model = template_linear_model(n, m)

""" Simulator stuff """
A_sim = 0.5 * np.array([[1., 0., 2., 0.],
                        [0., 1., 0., 1.],
                        [0., 0., 1., 2.],
                        [1., 0., 0., 1.],], **np_kwargs)
if m == 1:
    B_sim = np.array([[0.5],
                    [0.],
                    [0.],
                    [0.]], **np_kwargs)
elif m == 2:
    B_sim = np.array([[0.5, 0],
                    [0., 0],
                    [0., 0],
                    [0., 0.5]], **np_kwargs)
sim_p = {'A' : A_sim,
         'B' : B_sim}

Q_sim, R_sim = np.diag(np.ones(n, **np_kwargs)), np.diag(np.ones(m, **np_kwargs)) # np.ones((n,n)), np.ones((m,m))

simulator = template_linear_simulator(model, sim_p)
estimator = do_mpc.estimator.StateFeedback(model)

""" RL stuff """
gym.register(
    id="gymnasium_env/LQR-v0",
    entry_point=LQREnv,
)

env_kwargs = {k:v for k,v in zip(["n", "m", "Q", "R", "A", "B"], [n, m, Q_sim, R_sim, A_sim, B_sim])}
env = gym.make_vec("gymnasium_env/LQR-v0", num_envs=num_envs, max_episode_steps=10, **env_kwargs)

rb = ReplayBuffer(
    buffer_size=1000,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=kwargs['device'],
    handle_timeout_termination=False,
)

""" MPCritic stuff """
# numpy arrays share memory with corresponding pytorch model params
A_mpc = np.random.uniform(-1., 1., (n,n)).astype(np_kwargs['dtype']) if learn_dynamics else A_sim.copy()
B_mpc = np.random.uniform(-1., 1., (n,m)).astype(np_kwargs['dtype']) if learn_dynamics else B_sim.copy()
Q_mpc, R_mpc = Q_sim.copy(), R_sim.copy()
K_opt = calc_K(A_mpc, B_mpc, Q_mpc, R_mpc).astype(np_kwargs['dtype'])
P_opt = calc_P(A_mpc, B_mpc, Q_mpc, R_mpc).astype(np_kwargs['dtype'])
if mode == 'dpc':
    K = np.random.uniform(-1., 1., (m,n)).astype(np_kwargs['dtype'])
    P = P_opt
    mpc_mterm = QuadraticTerminalCost(n, P)
elif mode == 'critic':
    K = K_opt
    L = np.random.uniform(-1., 1., (n,n)).astype(np_kwargs['dtype'])
    mpc_mterm = PDQuadraticTerminalCost(n, L)
unc_p = {'A' : [A_mpc],
         'B' : [B_mpc]} # 1 uncertainty scenario considered

mpc_lterm = QuadraticStageCost(n, m, Q_mpc, R_mpc)
mpc_model = LinearDynamics(n, m, A_mpc, B_mpc)
dpc = LinearPolicy(n, m, K)
concat_dynamics = InputConcat(mpc_model)
dynamics = Dynamics(env=env, rb=rb, dx=concat_dynamics)

critic = MPCritic(model, dynamics, mpc_lterm, mpc_mterm, dpc, unc_p)
critic.mpc_settings['n_horizon'] = 1
critic.setup_mpc()
critic.requires_grad_(True)

""" Learning environment dynamics """
if learn_dynamics:
    obs, _ = env.reset()
    for _ in range(rb.buffer_size):
        action = dpc(torch.from_numpy(obs)).detach().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncated):
            if trunc:
                real_next_obs[idx] = info["final_observation"][idx]
        rb.add(obs, real_next_obs, action, reward, terminated, info)

        obs = next_obs

    params_sim, params_mpc = np.concat([A_sim, B_sim], axis=1), np.concat([A_mpc, B_mpc], axis=1)
    print(f"Before training: 'Distance' from true dynamics: {np.linalg.norm(params_sim - params_mpc, 'fro')}")
    dynamics.train()
    print(f"After training: 'Distance' from true dynamics: {np.linalg.norm(params_sim - params_mpc, 'fro')}")
    A_lrn = list(dynamics.best_model.values())[0].detach().numpy()
    B_lrn = list(dynamics.best_model.values())[1].detach().numpy()
    params_lrn = np.concat([A_sim, B_sim], axis=1)
    print(f"After training: 'Distance' from true dynamics: {np.linalg.norm(params_sim - params_lrn, 'fro')}")

"""
Because dynamics.best_model is a deepcopy, mpc_model, concat_dynamics, A_mpc, and B_mpc are not updated to reflect
"""

""" Neuromancer stuff """
batches = 5000
b = 256
problem = critic.setup_problem(mode)

if mode == 'dpc':
    """ Learning ficticious controller """
    train_data = DictDataset({'x': 3.*torch.randn(3333, 1, n)}, name='train')  # Split conditions into train and dev
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333, shuffle=False,
                                               collate_fn=train_data.collate_fn)
    
    dev_data = DictDataset({'x': 3.*torch.randn(3333, 1, n)}, name='dev')
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333, shuffle=False,
                                             collate_fn=dev_data.collate_fn)

    # optimizer and trainer
    dpc_optimizer = optim.Adam(critic.dpc.parameters(), lr=0.01)

    '''
    Two training loops presented: (A) Neuroamncer (B) Sample-based
        (A) Neuroamncer: Uses preexisting codes from Neuromancer to do training process.
                         Initial implementation using this produces an error of attempting
                         to backward twice through the graph (presumably the initial action
                         computed by the fictucious controller)
        (B) Sample-based: Sample-based learning strategy similar to deterministic policy
                          gradient, manipulating the ficticious controller to optimize the
                          critic's Q-function. If opting for this scheme, two system definitions
                          would not be required.
                          
    '''
    # # Training loop A
    # trainer = Trainer(
    #     problem,
    #     train_loader,
    #     dev_loader,
    #     dev_loader,
    #     optimizer=dpc_optimizer,
    #     epochs=1000,
    #     train_metric='train_loss',
    #     dev_metric='dev_loss',
    #     eval_metric='dev_loss',
    #     warmup=200,
    # )
    # best_model = trainer.train()
    # print(f"'Distance' from optimal gain: {np.linalg.norm(K_opt - K, 'fro')}")

    # Training loop B
    problem = critic.setup_problem(mode='critic')
    for i in range(batches):
        s = 3.*torch.randn((b, n), **kwargs)

        q_loss = -torch.mean(critic(s))

        dpc_optimizer.zero_grad()
        q_loss.backward()
        dpc_optimizer.step()

        if (i % 100) == 0:
            print(f"Iter {i}: 'Distance' from optimal gain: {np.linalg.norm(K_opt - K, 'fro')}")

    print(f'K == K^*:\n{np.isclose(K_opt, K)}')

elif mode == 'critic':
    """ Evaluating Q-function for random tuples, presumably coming from replay buffer """    
    # Optimizer and objective
    mse = torch.nn.MSELoss(reduction='mean')
    critic_params = list(critic.mpc_mterm.parameters()) # + list(stage_cost.parameters()) + list(linear_dynamics.parameters())
    critic_optimizer = optim.Adam(critic_params, lr=0.01)

    # Training loop
    for i in range(batches):

        s = 3.*torch.randn((b, n), **kwargs)
        with torch.no_grad():
            a = dpc(s) # actions of optimal policy
            r = -mpc_lterm(torch.concat((s,a), dim=-1)) # negative quadratic
            next_s = mpc_model(torch.concat((s,a), dim=-1)) # nominal dynamics

        # Steps for Q-learning!
        q_targ = r + critic(next_s)
        q_pred = critic(s, a)
        td_loss = mse(q_pred, q_targ)

        critic_optimizer.zero_grad()
        td_loss.backward()
        critic_optimizer.step()

        if (i % 100) == 0:
            P = L.T @ L + mpc_mterm.epsilon*np.diag(np.ones(n, **np_kwargs))
            print(f"Iter {i}: 'Distance' from optimal value function: {np.linalg.norm(P_opt - P, 'fro')}")

    print(f'P == P^*:\n{np.isclose(P_opt, P)}')