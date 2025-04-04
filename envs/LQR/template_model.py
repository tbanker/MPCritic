import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_model(n, m, symvar_type='MX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Certain parameters

    # States struct (optimization variables):
    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(n,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(m,1))

    # Fixed parameters:
    A = model.set_variable('_p', var_name='A', shape=(n,n))
    B = model.set_variable('_p', var_name='B', shape=(n,m))
    P = model.set_variable('_p', var_name='P', shape=(n,n))

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    model.set_expression(expr_name='stage_cost', expr=sum1(_x**2)+10*sum1(_u**2))
    model.set_expression(expr_name='terminal_cost', expr=bilin(P, _x))
    model.set_expression(expr_name='tracking_cost', expr=sum1(_x**2))

    # Differential equations (includes rescaling)
    model.set_rhs('x', A@_x+B@_u)

    # Build the model
    model.setup()

    return model
