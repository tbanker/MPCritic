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
        """ f(x,u) -> f(z) """
        # L4CasADi models must have 1 input (z), but Neuromancer needs 2 (x,u)
        z = torch.concat((x,u), dim=-1)
        return self.module(z)

""" Closed-loop 'system' """
concat_cost = InputConcat(critic.mpc_lterm)
stage_cost = Node(concat_cost, ['X', 'U'], ['l'], name='stage_cost')
terminal_cost = Node(critic.mpc_mterm, ['X'], ['V'], name='terminal_cost')

concat_dynamics = InputConcat(critic.mpc_model)
linear_dynamics = Node(concat_dynamics, ['X', 'U'], ['X'], name='linear_dynamics')
policy = Node(critic.dpc, ['X'], ['U'], name='policy')

# mpc_settings['n_horizon']+1 redundant terminal cost evaluations
# 1 redundant stage cost evaluations
# but objective matches that of do-mpc, horizon considered :)
cl_system = System([policy, stage_cost, linear_dynamics, terminal_cost])
cl_system.nsteps = critic.mpc_settings['n_horizon'] + 2

# random states as training data
train_data = DictDataset({'X': 3.*torch.randn(3333, 1, n)}, name='train')  # Split conditions into train and dev
dev_data = DictDataset({'X': 3.*torch.randn(3333, 1, n)}, name='dev')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333,
                                           collate_fn=train_data.collate_fn, shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333,
                                         collate_fn=dev_data.collate_fn, shuffle=False)

""" Objective and optimizer """
u = variable('U')
x = variable('X')
l = variable('l')
V = variable('V')

# Ignore last stage cost (k=mpc_settings['n_horizon']+1), only consider last temrinal cost (k=mpc_settings['n_horizon']+2).
# Stage cost should be weighted (mpc_settings['n_horizon']+1) times more than terminal cost.
# This is because Neuromancer computes mean(stage costs) rather than mean(sum of stage_costs over horizon) for this formulation.
stage_loss = Objective(var=(critic.mpc_settings['n_horizon']+1.)*l[:, :-1, :], name="stage_loss")
terminal_loss = Objective(var=(V[:, [-1], :]), name="terminal_loss")
loss = PenaltyLoss([stage_loss, terminal_loss], [])

problem = Problem([cl_system], loss)
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)

if learn:
    """ Learning ficticious controller with prediction horizon of 10 """
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        dev_loader,
        optimizer=optimizer,
        epochs=1000,
        train_metric="train_loss",
        dev_metric="dev_loss",
        eval_metric='dev_loss',
        warmup=200,
    )

    best_model = trainer.train()
    K_opt = calc_K(A_mpc, B_mpc, Q, R).astype(np_kwargs['dtype'])
    print(f"'Distance' from optimal gain: {np.linalg.norm(K_opt - K, 'fro')}")

else:
    """ Evaluating loss function for random inputs """
    loss_dict = {k: torch.randn(20, cl_system.nsteps, 1) for k in loss.input_keys}
    out_dict = loss(loss_dict)

    """ Evaluating Q-function for random tuples, presumably coming from replay buffer """
    eval_data = DictDataset({"X" : 3.*torch.randn(1024, 1, n),
                             "U"  : 3.*torch.randn(1024, 1, m),
                             "R"  : 3.*torch.randn(1024, 1, 1),
                             "X'" : 3.*torch.randn(1024, 1, n),}, name='train')
    eval_buffer = torch.utils.data.DataLoader(eval_data, batch_size=64,
                                            collate_fn=train_data.collate_fn, shuffle=False)

    def get_subdict(dict, keys, rekey=False, val=None):
        """ Extract sub-dictionary of dict & rekey either ["X'"] or val to ["X"] """
        subdict = {key : dict[key] for key in keys}
        if rekey and (val != None):
            subdict["X"] = val
        elif rekey:
            subdict["X"] = subdict.pop("X'")
        return subdict

    """ Bad steps for Q-learning! """
    critic.requires_grad_(True)
    for batch in eval_buffer:
        with torch.no_grad():
            # cost of ficticious controller rollout for cl_system.nsteps from s'
            q_x_next = problem(get_subdict(batch, ["X'", "name"], rekey=True))['train_loss']
            # bad target Q-value for (r, s')
            q_targ = torch.mean(batch['R']) + q_x_next # ||x+y|| <= ||x|| + ||y|| :(

        # stage cost of (s,a)
        z = torch.concat((batch['X'], batch['U']), dim=-1).squeeze(1)
        q_l = torch.mean(critic.mpc_lterm(z))

        # cost of ficticious controller rollout for (cl_system.nsteps-1) from predicted s'
        cl_system.nsteps -= 1
        x_pred = critic.mpc_model(z).unsqueeze(1)
        q_x = problem(get_subdict(batch, ["X", "name"], rekey=True, val=x_pred))['train_loss']
        # bad predicted Q-value for (s, a)
        q_pred = q_l + q_x # ||x+y|| <= ||x|| + ||y|| :(

        cl_system.nsteps += 1