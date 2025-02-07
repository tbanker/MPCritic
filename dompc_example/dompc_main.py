import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

from dompc_templates import template_linear_model
from dompc_templates import template_linear_simulator
from dompc_templates import template_LQR_mpc

""" User settings: """
show_animation = True
store_results = False

"""
Get configured do-mpc modules.:
"""
model = template_linear_model()
simulator = template_linear_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)
mpc = template_LQR_mpc(model)


"""
Set initial state
"""
np.random.seed(99)

e = np.ones([model.n_x,1])
x0 = np.random.uniform(-3*e,3*e) # Values between +3 and +3 for all states
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()

"""
Setup graphic:
"""

fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
plt.ion()

"""
Run MPC main loop:
"""

for k in range(2):
    u0 = mpc.make_step(x0)
    print(x0, u0)
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
    do_mpc.data.save_results([mpc, simulator], 'LQR')
