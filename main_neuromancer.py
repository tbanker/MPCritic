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

from modules.templates import template_linear_model, template_linear_simulator
from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, LinearDynamics, LinearPolicy
from modules.mpcritic import MPCritic
from modules.utils import calc_K, calc_P

from neuromancer.dataset import DictDataset
from neuromancer.trainer import Trainer

""" User settings: """
mode = 'dpc' # 'dpc' or 'critic'
np_kwargs = {'dtype' : np.float32}
kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}
seed = 43
np.random.seed(seed)
torch.manual_seed(seed)

""" Get configured do-mpc modules: """
b, n, m = 1, 4, 2
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

simulator = template_linear_simulator(model, sim_p)
estimator = do_mpc.estimator.StateFeedback(model)

""" MPC stuff """
# numpy arrays share memory with corresponding pytorch model params
A_mpc = A_sim.copy() # + np.random.uniform(-0.4, 0.4, A_sim.shape).astype(np_kwargs['dtype'])
B_mpc = B_sim.copy() # + np.random.uniform(-0.4, 0.4, B_sim.shape).astype(np_kwargs['dtype'])
Q, R = np.diag(np.ones(n, **np_kwargs)), np.diag(np.ones(m, **np_kwargs)) # np.ones((n,n)), np.ones((m,m))
K_opt = calc_K(A_mpc, B_mpc, Q, R).astype(np_kwargs['dtype'])
P_opt = calc_P(A_mpc, B_mpc, Q, R).astype(np_kwargs['dtype'])
if mode == 'dpc':
    K = np.random.uniform(-1., 1., (m,n)).astype(np_kwargs['dtype'])
    P = P_opt
elif mode == 'critic':
    K = K_opt
    P = np.random.uniform(-1., 1., (n,n)).astype(np_kwargs['dtype'])
unc_p = {'A' : [A_mpc],
         'B' : [B_mpc]} # 1 uncertainty scenario considered
mpc_lterm = QuadraticStageCost(n, m, Q, R)
mpc_mterm = QuadraticTerminalCost(n, P)
mpc_model = LinearDynamics(n, m, A_mpc, B_mpc)
dpc = LinearPolicy(n, m, K)

critic = MPCritic(model, mpc_model, mpc_lterm, mpc_mterm, dpc, unc_p)
critic.mpc_settings['n_horizon'] = 1
critic.setup_mpc()
critic.requires_grad_(True)

""" Neuromancer stuff """
batches = 3000
b = 64
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

    print(K_opt - K)

elif mode == 'critic':
    """ Evaluating Q-function for random tuples, presumably coming from replay buffer """    
    # Optimizer and objective
    mse = torch.nn.MSELoss(reduction='mean')
    critic_params = list(critic.mpc_mterm.parameters()) # + list(stage_cost.parameters()) + list(linear_dynamics.parameters())
    critic_optimizer = optim.Adam(critic_params, lr=0.1)

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
            print(f"Iter {i}: 'Distance' from optimal value function: {np.linalg.norm(P_opt - P, 'fro')}")

    print(P_opt - P)