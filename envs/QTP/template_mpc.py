import numpy as np
import casadi as cs
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

import l4casadi as l4c

def template_mpc(model, goal = np.array([[0.4],[0.4]]), mpc_mode = "nstep_vfmpc", n_horizon = 20, silence_solver = True, solver="pardiso",
                RL_env=False, uncertain_params = "nominal"):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    # Set settings of MPC:
    mpc.settings.n_robust = 1 # set to 0 if you just need a quick proof of concept
    mpc.settings.open_loop = False
    mpc.settings.t_step = 1.
    mpc.settings.state_discretization = 'collocation'
    mpc.settings.collocation_type = 'radau'
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 1
    mpc.settings.store_full_solution = True
    mpc.use_terminal_bounds = True

    if silence_solver:
        mpc.settings.supress_ipopt_output()
    # see https://coin-or.github.io/Ipopt/OPTIONS.html and https://arxiv.org/pdf/1909.08104
    mpc.nlpsol_opts = {'ipopt.linear_solver': solver} # pardiso, MA27, MA57, spral, HSL_MA86, mumps (mumps is do-mpc default; they also recommend MA27 for a speedup)

    mpc.bounds['lower','_u','v_1'] = 0.1 # 0.
    mpc.bounds['lower','_u','v_2'] = 0.1 # 0.

    mpc.bounds['upper','_u','v_1'] = 10. # 3.26
    mpc.bounds['upper','_u','v_2'] = 10. # 4.

    if uncertain_params == "include_truth": # the default setting from do-mpc's examples
        gamma_1_var, gamma_2_var = [np.array([0.2, 0.15, 0.25]), np.array([0.2, 0.15, 0.25])]
    if uncertain_params == "nominal": 
        gamma_1_var, gamma_2_var = [np.array([0.2]), np.array([0.2])]

    mpc.set_uncertainty_values(gamma_1 = gamma_1_var, gamma_2 = gamma_2_var)

    if mpc_mode == "baseline" or mpc_mode == "nominal":
        mpc.settings.n_horizon = n_horizon # 20 is the original value
        if mpc_mode == "nominal":
            mpc.settings.n_robust = 0

        mpc.bounds['lower', '_x', 'h_1s'] = 0. # 0.2
        mpc.bounds['lower', '_x', 'h_2s'] = 0. # 0.2
        mpc.bounds['upper', '_x', 'h_1s'] = 0.6 # 1.36
        mpc.bounds['upper', '_x', 'h_2s'] = 0.6 # 1.36

        lterm = sum1((vertcat(model.x['h_1s'], model.x['h_2s']) - goal)**2)
        mterm = sum1((vertcat(model.x['h_1s'], model.x['h_2s']) - goal)**2)
        mpc.set_objective(mterm=mterm, lterm=lterm)

        mpc.set_rterm(v_1 = 1e-2, v_2 = 1e-2)

        # Instead of having a regular bound on T_R:
        # We can also have soft constraints as part of the set_nl_cons method:

        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_now):
            tvp_template['_tvp',:, 'goal'] = goal
            return tvp_template
        mpc.set_tvp_fun(tvp_fun)
        
        mpc.setup()

    return mpc