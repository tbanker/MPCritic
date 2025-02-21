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

from neuromancer.system import Node, System
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable, Variable, Objective
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer

""" User settings: """
mode = "Critic" # "DPC" or "Critic"
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
K_opt = calc_K(A_mpc, B_mpc, Q, R).astype(np_kwargs['dtype'])
P_opt = calc_P(A_mpc, B_mpc, Q, R).astype(np_kwargs['dtype'])
if mode == "DPC":
    K = np.random.uniform(-1., 1., (m,n)).astype(np_kwargs['dtype'])
    P = P_opt
elif mode == "Critic":
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
critic.template_mpc()
critic.requires_grad_(True)

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

""" 'System' nodes """
concat_cost = InputConcat(critic.mpc_lterm)
stage_cost = Node(concat_cost, ['X', 'U'], ['l'], name='stage_cost')
terminal_cost = Node(critic.mpc_mterm, ['X'], ['V'], name='terminal_cost')

concat_dynamics = InputConcat(critic.mpc_model)
linear_dynamics = Node(concat_dynamics, ['X', 'U'], ['X'], name='linear_dynamics')
policy = Node(critic.dpc, ['X'], ['U'], name='policy')

""" Objective """
u = variable('U')
x = variable('X')
l = variable('l')
V = variable('V')

# Ignore last stage cost (k=mpc_settings['n_horizon']+1), only consider last temrinal cost (k=mpc_settings['n_horizon']+2).
# Stage cost should be weighted (mpc_settings['n_horizon']+1) times more than terminal cost.
# This is because Neuromancer computes mean(stage costs) rather than mean(sum of stage_costs over horizon) for this formulation.
stage_loss = Objective(var=(critic.mpc_settings['n_horizon']+1.)*l[:, :-1, :], name="stage_loss")
terminal_loss = Objective(var=(V[:, [-1], :]), name="terminal_loss")
dpc_loss = PenaltyLoss([stage_loss, terminal_loss], [])

if mode == "DPC":
    """ Learning ficticious controller """
    # mpc_settings['n_horizon']+1 redundant terminal cost evaluations
    # 1 redundant stage cost evaluations
    # but objective matches that of do-mpc, horizon considered :)
    cl_system = System([policy, stage_cost, linear_dynamics, terminal_cost],
                        nsteps = critic.mpc_settings['n_horizon'] + 2)
    problem = Problem([cl_system], dpc_loss)

    # random states as training data
    train_data = DictDataset({'X': 3.*torch.randn(3333, 1, n)}, name='train')  # Split conditions into train and dev
    dev_data = DictDataset({'X': 3.*torch.randn(3333, 1, n)}, name='dev')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=3333, shuffle=False,
                                            collate_fn=train_data.collate_fn)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=3333, shuffle=False,
                                            collate_fn=dev_data.collate_fn)

    # optimizer and trainer
    dpc_optimizer = optim.Adam(policy.parameters(), lr=0.01)
    trainer = Trainer(
        problem,
        train_loader,
        dev_loader,
        dev_loader,
        optimizer=dpc_optimizer,
        epochs=1000,
        train_metric="train_loss",
        dev_metric="dev_loss",
        eval_metric='dev_loss',
        warmup=200,
    )

    # run training
    best_model = trainer.train()
    print(f"'Distance' from optimal gain: {np.linalg.norm(K_opt - K, 'fro')}")

elif mode == "Critic":
    """ Evaluating Q-function for random tuples, presumably coming from replay buffer """

    class EvalSystem(System):
        """ System for evaluation; accepts input_dicts containing (X) or (X, U, predicted X') """
        def init(self, data):
            """ Compute X', l, and V if given U; otherwise, return X """
            if "U" in data.keys():
                # evaluate all nodes for (s,a) EXCEPT for policy node (first node)
                for node in self.nodes[1:]:
                    indata = {k: data[k][:, 0] for k in node.input_keys}
                    outdata = node(indata) 
                    data = self.cat(data, outdata)
                return data
            else:
                return data
        
        def forward(self, input_dict):
            """ Rollout ficticious controller from latest (x) """
            data = input_dict.copy()
            nsteps = self.nsteps if self.nsteps is not None else data[self.nstep_key].shape[1]
            data = self.init(data)
            # adapt number of steps in rollout depending on length of (x)
            for i in range(data['X'].shape[1]-1,nsteps):
                for node in self.nodes:
                    indata = {k: data[k][:, i] for k in node.input_keys}
                    outdata = node(indata)
                    data = self.cat(data, outdata)
            return data
        
    def get_subdict(dictdataset, index, keys, rekey=False, val=None):
        """ Extract sub-dictionary of dictdataset @ index & rekey either ["X'"] or val to ["X"] """
        subdict = dictdataset.__getitem__(index)
        subdict = {k: v.unsqueeze(0) for k, v in subdict.items()}
        # Neuromancer problems expect a "Dict" with key "name" and method "copy"
        subdict = {**subdict, "name" : dictdataset.name}

        subdict = {key : subdict[key] for key in (keys + ["name"])}
        if rekey and (val != None):
            subdict["X"] = val
        elif rekey:
            subdict["X"] = subdict.pop("X'")
        return subdict
    
    cl_system = EvalSystem([policy, stage_cost, linear_dynamics, terminal_cost],
                       nsteps = critic.mpc_settings['n_horizon'] + 2,)
    problem = Problem([cl_system], dpc_loss)

    # Optimizer and objective
    mse = torch.nn.MSELoss(reduction="mean")
    critic_params = list(terminal_cost.parameters()) # + list(stage_cost.parameters()) + list(linear_dynamics.parameters())
    critic_optimizer = optim.Adam(critic_params, lr=1.)
    batches = 1000
    b = 64

    # Training loop
    for _ in range(batches):

        # random state-action pairs as training data
        s = 3.*torch.randn((b, n), **kwargs)
        with torch.no_grad():
            a = dpc(s) # actions of optimal policy
            r = -concat_cost(s,a) # negative quadratic
            next_s = concat_dynamics(s,a) # nominal dynamics

        eval_data = DictDataset({"X"  : s.unsqueeze(1),
                                "U"  : a.unsqueeze(1),
                                "R"  : r.unsqueeze(1),
                                "X'" : next_s.unsqueeze(1),}, name='train')

        # """ Critic Validation """
        # # Compare Q-values for (s) and (s,a) initializations of closed-loop system
        # q_init_s = torch.empty((b,1), **kwargs)
        # q_init_sa = torch.empty((b,1), **kwargs)

        # for i in range(eval_data.length):
        #     # Neuromancer problems expect a "Dict" with key "name" and method "copy"
        #     eval_data_i = eval_data.__getitem__(i)
        #     eval_data_i = {k: v.unsqueeze(0) for k, v in eval_data_i.items()}
        #     eval_data_i = {**eval_data_i, "name" : eval_data.name}

        #     s_data = get_subdict(eval_data_i, ["X", "name"])
        #     sa_data = get_subdict(eval_data_i, ["X", "U", "name"])
        #     # make "U" to equal that of closed-loop system
        #     sa_data["U"] = policy(s_data)["U"]

        #     q_init_s[i,:] = -problem(s_data)['train_loss']
        #     q_init_sa[i,:] = -problem(sa_data)['train_loss']

        # print(f"Q(init(s)) = Q(init(s,a)): {torch.allclose(q_init_s, q_init_sa)}")

        """ Steps for Q-learning! """
        q_pred = torch.empty((b,1), **kwargs)
        q_targ = torch.empty((b,1), **kwargs)

        # Get Q predictions and targets
        for i in range(eval_data.length):
            next_s_data = get_subdict(eval_data, i, ["X'"], rekey=True)
            sa_data = get_subdict(eval_data, i, ["X", "U"])
            r_data = get_subdict(eval_data, i, ["R"])

            with torch.no_grad():
                # cost of ficticious controller rollout for cl_system.nsteps from s'
                q_next_s = -problem(next_s_data)['train_loss']
                # target Q-value for (r, s')
                q_targ[i,:] = r_data['R'] + q_next_s

            # cost of (s,a) followed by ficticious controller rollout for (cl_system.nsteps-1)
            q_pred[i,:] = -problem(sa_data)['train_loss']
        
        td_loss = mse(q_pred, q_targ)

        critic_optimizer.zero_grad()
        td_loss.backward()
        critic_optimizer.step()

        print(f"'Distance' from optimal value function: {np.linalg.norm(P_opt - P, 'fro')}")

    print(P_opt - P)