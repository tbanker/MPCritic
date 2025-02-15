import numpy as np
import torch
from torch import optim
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

from neuromancer.system import Node, System
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable, Variable, Objective
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer

""" User settings: """
learn = True
show_animation = True
store_results = False
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
K = np.random.uniform(-1., 1., (m,n)).astype(np_kwargs['dtype']) # calc_K(A_mpc, B_mpc, Q, R).astype(np_kwargs['dtype']) # np.ones((m,n), **np_kwargs)
P = calc_P(A_mpc, B_mpc, Q, R).astype(np_kwargs['dtype'])
unc_p = {'A' : [A_mpc],
         'B' : [B_mpc]} # 1 uncertainty scenario considered
mpc_lterm = QuadraticStageCost(n, m, Q, R)
mpc_mterm = QuadraticTerminalCost(n, P)
mpc_model = LinearDynamics(n, m, A_mpc, B_mpc)
dpc = LinearPolicy(n, m, K)

critic = MPCritic(model, mpc_model, mpc_lterm, mpc_mterm, dpc, unc_p)
critic.template_mpc()
critic.dpc.requires_grad_(True)
critic.mpc_model.requires_grad_(True)

""" Neuromancer stuff """
class InputConcat(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, u):
        """ f(z) <- f(x,u) """
        # L4CasADi models must have 1 input (z), but Neuromancer needs 2 (x,u)
        z = torch.concat((x,u), dim=-1)
        return self.module(z)

# closed-loop system
concat_dynamics = InputConcat(critic.mpc_model)
concat_cost = InputConcat(critic.mpc_lterm)
stage_cost = Node(concat_cost, ['X', 'U'], ['l'], name='stage_cost')
terminal_cost = Node(critic.mpc_mterm, ['X'], ['V'], name='terminal_cost')

linear_dynamics = Node(concat_dynamics, ['X', 'U'], ['X'], name='linear_dynamics')
policy = Node(critic.dpc, ['X'], ['U'], name='policy')
cl_system = System([policy, stage_cost, linear_dynamics, terminal_cost])

# random states as training data
train_data = DictDataset({'X': 3.*torch.randn(3333, 1, n)}, name='train')  # Split conditions into train and dev
dev_data = DictDataset({'X': 3.*torch.randn(3333, 1, n)}, name='dev')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                         collate_fn=dev_data.collate_fn, shuffle=False)

# objective and optimizer
u = variable('U')
x = variable('X')
l = variable('l')
V = variable('V')

stage_loss = 1. * (l[:, :-1, :] == 0.)
terminal_loss = 1. * (V[:, [-1], :] == 0.)
stage_loss.name = "stage_loss"
terminal_loss.name = "terminal_loss"

loss = PenaltyLoss([stage_loss, terminal_loss], [])
problem = Problem([cl_system], loss)
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)

# learning
trainer = Trainer(
    problem,
    train_loader,
    dev_loader,
    dev_loader,
    optimizer=optimizer,
    epochs=400,
    train_metric="train_loss",
    dev_metric="dev_loss",
    eval_metric='dev_loss',
    warmup=400,
)

# Train model with prediction horizon of 2
cl_system.nsteps = 2
best_model = trainer.train()