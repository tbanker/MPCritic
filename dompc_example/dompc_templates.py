import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from scipy.linalg import solve_discrete_are

def template_LQR_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    # Fixed parameters:
    A = model.set_variable('_p', var_name='A', shape=(4,4))
    B = model.set_variable('_p', var_name='B', shape=(4,1))
    P = model.set_variable('_p', var_name='P', shape=(4,4))

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    model.set_expression(expr_name='stage_cost', expr=sum1(_x**2)+sum1(_u**2))
    model.set_expression(expr_name='terminal_cost', expr=bilin(P, _x))

    x_next = model.set_variable(var_type='_z', var_name='x_next', shape=(4,1))

    model.set_rhs('x', x_next)

    model.set_alg('x_next', x_next-A@_x-B@_u)

    model.setup()

    return model

def template_linear_simulator(model):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 1.0)

    p_num = simulator.get_p_template()
    p_num['A'] = 0.5 * np.array([[1., 0., 2., 0.],
                                 [0., 1., 0., 1.],
                                 [0., 0., 1., 2.],
                                 [1., 0., 0., 1.],])  # (n, n)
    p_num['B'] = np.array([[0.5],
                           [0.],
                           [0.],
                           [0.]]) # (n, m)

    def p_fun(t_now):
        return p_num

    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator

def template_LQR_mpc(model, silence_solver=False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    mpc.settings.n_robust = 0
    mpc.settings.n_horizon = 5
    mpc.settings.t_step = 1.0
    mpc.settings.store_full_solution =True

    if silence_solver:
        mpc.settings.supress_ipopt_output()

    lterm = model.aux['stage_cost']
    mterm = model.aux['terminal_cost']

    mpc.set_objective(lterm=lterm, mterm=mterm)
    mpc.set_rterm(u=0.)

    A = 0.5 * np.array([[1., 0., 2., 0.],
                        [0., 1., 0., 1.],
                        [0., 0., 1., 2.],
                        [1., 0., 0., 1.],])
    B = np.array([[0.5],
                  [0.],
                  [0.],
                  [0.]])
    Q = np.diag(np.ones(A.shape[0]))
    R = np.diag(np.ones(B.shape[1]))
    P = solve_discrete_are(A,B,Q,R)

    mpc.set_uncertainty_values(A = [A], B = [B], P = [P])

    mpc.setup()

    return mpc