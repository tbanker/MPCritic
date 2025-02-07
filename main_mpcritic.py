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
A_mpc = A_sim.copy() + np.random.uniform(-0.4, 0.4, A_sim.shape).astype(np_kwargs['dtype'])
B_mpc = B_sim.copy() # + np.random.uniform(-0.4, 0.4, B_sim.shape).astype(np_kwargs['dtype'])
Q, R = np.diag(np.ones(n, **np_kwargs)), np.diag(np.ones(m, **np_kwargs)) # np.ones((n,n)), np.ones((m,m))
K = calc_K(A_mpc, B_mpc, Q, R).astype(np_kwargs['dtype']) # np.ones((m,n), **np_kwargs)
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

""" Set initial state """
e = np.ones([model.n_x,1])
x0 = np.random.uniform(-3*e,3*e) # Values between +3 and +3 for all states
simulator.x0 = x0
estimator.x0 = x0
critic.init_mpc(x0)

""" Setup graphic: """
fig, ax, graphics = do_mpc.graphics.default_plot(critic.mpc.data)
plt.ion()

""" Run MPC main loop: """
b = 10
T = 5*b
critic_optimizer = optim.SGD([critic.mpc_model.A], lr=0.01)
dpc_optimizer = optim.SGD(critic.dpc.parameters(), lr=0.01)
critic_loss, dpc_loss = 0., 0.

s = torch.tensor(x0.T, **kwargs)
for t in range(T):
    u0 = critic.mpc.make_step(x0)

    a = torch.tensor(u0.T, **kwargs)
    r = - torch.sum(s**2) - torch.sum(a**2)

    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    dpc_loss += torch.norm(critic.dpc(s) - a)**2 / b

    q = critic(s,a)
    s = torch.tensor(x0.T, **kwargs)
    critic_loss += (q - (r + critic(s,critic.dpc(s))))**2 / b

    if show_animation:
        graphics.plot_results(t_ind=t)
        graphics.plot_predictions(t_ind=t)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

    if learn and ((t+1) % b) == 0:
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        critic_loss = 0.
        
        dpc_optimizer.zero_grad()
        dpc_loss.backward()
        dpc_optimizer.step()
        dpc_loss = 0.

        # K = calc_K(A_mpc, B_mpc, Q, R).astype(np_kwargs['dtype']) # np.ones((m,n), **np_kwargs)
        # critic.dpc = LinearPolicy(n, m, K)

        A_err = np.linalg.norm(A_sim-A_mpc, ord='fro')
        B_err = np.linalg.norm(B_sim-B_mpc, ord='fro')
        print(f"A Err: {A_err}, B Err: {B_err}")

        critic.online_update(x0) # turns requires_grad off
        critic.dpc.requires_grad_(True)
        critic.mpc_model.requires_grad_(True)

        graphics.data = critic.mpc.data


input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([critic.mpc, simulator], 'LQR')