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


def template_simulator(model, uncertain_params = "nominal", goal = np.array([[0.4],[0.4]])):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 1.
    }

    simulator.set_param(**params_simulator)

    tvp_template = simulator.get_tvp_template()
    def tvp_fun(t_now):
        tvp_template['goal'] = goal
        return tvp_template
    simulator.set_tvp_fun(tvp_fun)

    p_num = simulator.get_p_template()
    if uncertain_params == "include_truth": # the default option from do-mpc's examples
        gamma_1_var, gamma_2_var = [np.array([0.2, 0.15, 0.25]), np.array([0.2, 0.15, 0.25])]
    elif uncertain_params == "nominal":
        gamma_1_var, gamma_2_var = [np.array([0.2]), np.array([0.2])]
    def p_fun(t_now):
        gamma_1 = np.random.choice(gamma_1_var)
        gamma_2 = np.random.choice(gamma_2_var)
        p_num['gamma_1'], p_num['gamma_2'] = gamma_1, gamma_2
        return p_num 

    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator