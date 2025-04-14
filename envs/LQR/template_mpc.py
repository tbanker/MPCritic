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

from scipy.linalg import solve_discrete_are

def template_mpc(model, goal = None, mpc_mode = "nominal", n_horizon = 5, silence_solver = True, solver="pardiso",
                RL_env=False, uncertain_params = "nominal"):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    n, m = model.n_x, model.n_u
    mpc = do_mpc.controller.MPC(model)

    # Set settings of MPC:
    mpc.settings.n_robust = 0
    mpc.settings.open_loop = False
    mpc.settings.t_step = 1.0
    mpc.settings.store_full_solution = True

    if silence_solver:
        mpc.settings.supress_ipopt_output()
        
    mpc.nlpsol_opts = {'ipopt.linear_solver': solver}

    mpc.bounds['lower','_u','u'] = -np.ones(m)
    mpc.bounds['upper','_u','u'] = np.ones(m)

    if uncertain_params == "nominal": 
        # assumes n=m
        A = (np.diag(1.01*np.ones(n)) + np.diag(0.01*np.ones(n-1), k=1) + np.diag(0.01*np.ones(n-1), k=-1)).astype(np.float32)
        B = np.diag(np.ones(m)).astype(np.float32)
        P = solve_discrete_are(A, B, q=np.diag(np.ones(n)), r=10*np.diag(np.ones(m)))

        mpc.set_uncertainty_values(A = [A], B = [B], P = [P])

    if mpc_mode == "baseline" or mpc_mode == "nominal":
        mpc.settings.n_horizon = n_horizon
        if mpc_mode == "nominal":
            mpc.settings.n_robust = 0

        mpc.bounds['lower','_x','x'] = -np.ones(n)
        mpc.bounds['upper','_x','x'] = np.ones(n)

        lterm = model.aux['stage_cost']
        mterm = model.aux['terminal_cost']
        mpc.set_objective(mterm=mterm, lterm=lterm)

        mpc.set_rterm(u=0.)
        
        mpc.setup()

    return mpc