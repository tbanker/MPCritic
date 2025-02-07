import numpy as np
import casadi as ca
from casadi import tools
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

def template_linear_model(n, m, symvar_type='MX'):
    """ template_model: Variables / RHS / AUX """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(n,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(m,1))

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    model.set_expression(expr_name='stage_cost', expr=ca.sum1(_x**2)+ca.sum1(_u**2))
    model.set_expression(expr_name='terminal_cost', expr=ca.sum1(_x**2))
   
    # Fixed parameters:
    A = model.set_variable('_p', var_name='A', shape=(n,n))
    B = model.set_variable('_p', var_name='B', shape=(n,m))

    x_next = model.set_variable(var_type='_z', var_name='x_next', shape=(n,1))

    model.set_rhs('x', x_next)

    model.set_alg('x_next', x_next-A@_x-B@_u)

    model.setup()

    return model

def template_linear_simulator(model, sim_p):
    """ template_optimizer: tuning parameters """
    simulator = do_mpc.simulator.Simulator(model)

    simulator.set_param(t_step = 1.0)

    p_num = simulator.get_p_template()
    for name, value in sim_p.items():
        p_num[name] = value

    def p_fun(t_now):
        return p_num

    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator