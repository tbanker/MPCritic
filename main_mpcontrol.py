import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

from modules.templates import template_linear_model, template_linear_simulator
from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, LinearDynamics, LinearPolicy
from modules.mpcritic import MPCritic


""" User settings: """
show_animation = True
store_results = False
np_kwargs = {'dtype' : np.float32}

""" Get configured do-mpc modules: """
b, n, m = 1, 4, 2
model = template_linear_model(n, m)

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

Q, R = np.diag(np.ones(n, **np_kwargs)), np.diag(np.ones(m, **np_kwargs)) # np.ones((n,n)), np.ones((m,m))
A_mpc, B_mpc = A_sim.copy(), B_sim.copy() # np.ones((n,n)), np.ones((n,m))
K = np.ones((m,n), **np_kwargs)
unc_p = {'A' : [A_mpc],
         'B' : [B_mpc]} # uncertainty scenarios
mpc_lterm = QuadraticStageCost(n, m, Q, R)
mpc_mterm = QuadraticTerminalCost(n, Q)
mpc_model = LinearDynamics(n, m, A_mpc, B_mpc)
dpc = LinearPolicy(n, m, K)

critic = MPCritic(model, mpc_model, mpc_lterm, mpc_mterm, dpc, unc_p)
critic.setup_mpc()

""" Set initial state """
np.random.seed(99)
torch.manual_seed(99)

e = np.ones([model.n_x,1])
x0 = np.random.uniform(-3*e,3*e) # Values between +3 and +3 for all states
simulator.x0 = x0
estimator.x0 = x0
critic.init_mpc(x0)

""" Setup graphic: """
fig, ax, graphics = do_mpc.graphics.default_plot(critic.mpc.data)
plt.ion()

""" Run MPC main loop: """
for k in range(50):
    u0 = critic.mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([critic.mpc, simulator], 'LQR')
