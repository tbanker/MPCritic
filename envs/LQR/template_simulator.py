import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
import copy


def template_simulator(model, uncertain_params="nominal", goal=None):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    n, m = model.n_x, model.n_u
    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 1.0)

    p_num = simulator.get_p_template()
    if uncertain_params == "nominal":
        A_var = [(np.diag(1.01*np.ones(n)) + np.diag(0.01*np.ones(n-1), k=1) + np.diag(0.01*np.ones(n-1), k=-1)).astype(np.float32)]
        B_var = [np.diag(np.ones(m)).astype(np.float32)]
    def p_fun(t_now):
        A = A_var[np.random.choice(np.arange(len(A_var)))]
        B = B_var[np.random.choice(np.arange(len(B_var)))]
        p_num['A'], p_num['B'] = A, B
        return p_num

    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator