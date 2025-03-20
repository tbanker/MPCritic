import time
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from casadi.tools import *
import pdb
import sys
import time
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.differentiator import DoMPCDifferentiator

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
mpc_diff = DoMPCDifferentiator(mpc)

mpc_diff._get_Lagrangian_sym()
lagr = mpc_diff.L_sym
nlp, nlpbounds = mpc_diff._get_do_mpc_nlp()
x = nlp['x']
f = nlp['f']
g = nlp['g']
p = nlp['p']
lam_g = ca.SX.sym("lam_g", mpc_diff.n_g, 1) # ca.SX.sym("lam_x", x.sparsity())
lam_x = ca.SX.sym("lam_x", mpc_diff.n_x, 1) # ca.SX.sym("lam_g", g.sparsity())
lam = ca.vertcat(lam_g, lam_x)
z = ca.vertcat(x, lam)

# A_sym = ca.hessian(lagr, z)[0]
# A_func = ca.Function("A", [z,p], [A_sym], ["z_opt", "p_opt"], ["A"])
# B_sym = ca.jacobian(ca.gradient(lagr,z),p)
# B_func = ca.Function("B", [z,p], [B_sym], ["z_opt", "p_opt"], ["B"])
# gradz_lagr = ca.gradient(lagr, z)[0]
# gradz_lagr_func = ca.Function("gradz_lagr", [z,p], [gradz_lagr], ["z_opt", "p_opt"], ["gradz_lagr"])

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
    start = time.time()
    u0 = mpc.make_step(x0)
    forward_time = time.time() - start

    start = time.time()
    dx_dp_num, dlam_dp_num = mpc_diff.differentiate()
    backward_time = time.time() - start

    print(f"Forward: {forward_time}\n Backward: {backward_time}")

    nlp_sol = mpc_diff._get_do_mpc_nlp_sol()
    x_num = nlp_sol['x']
    lam_num = ca.vertcat(nlp_sol['lam_g'], nlp_sol['lam_x'])
    z_num = ca.vertcat(x_num, lam_num)
    p_num = mpc_diff._get_p_num()
    # A_num = A_func(z_num, p_num)
    
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