import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


def template_model(symvar_type='MX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Certain parameters
    # https://cse.sc.edu/~gatzke/publications/00pse3.pdf
    g = 9.81 # Gravity [m/s^2]
    A_1 = 1. # 0.06 # Tank 1 cross section area [m^2]
    A_2 = 1. # 0.06 # Tank 2 cross section area [m^2]
    A_3 = 1. # 0.06 # Tank 3 cross section area [m^2]
    A_4 = 1. # 0.06 # Tank 4 cross section area [m^2]
    a_1 = 35 # 1.31  # Tank 1 outlet cross section area [cm^2]
    a_2 = 30 # 1.51  # Tank 2 outletcross section area [cm^2]
    a_3 = 20 # 0.927 # Tank 3 outlet cross section area [cm^2]
    a_4 = 25 # 0.882 # Tank 4 outlet cross section area [cm^2]
    # gamma_1 = 0.2 # 0.3 # Fraction Pump 1 flow bypassed into Tank 1 (otherwise into Tank 4) []
    # gamma_2 = 0.2 # 0.4 # Fraction Pump 2 flow bypassed into Tank 2 (otherwise into Tank 3) []
    k_1 = 0.00085 # 1. # Pump 1 gain [m^3/V.s]
    k_2 = 0.00095 # 1. # Pump 2 gain [m^3/V.s]

    # time-varying parameters:
    goal = model.set_variable('_tvp', 'goal', shape=(2,1))

    # States struct (optimization variables):
    h_1s = model.set_variable(var_type='_x',  var_name='h_1s', shape=(1,1))  # Tank 1 fill level [m]
    h_2s = model.set_variable(var_type='_x',  var_name='h_2s', shape=(1,1))  # Tank 2 fill level [m]
    h_3s = model.set_variable(var_type='_x',  var_name='h_3s', shape=(1,1))  # Tank 3 fill level [m]
    h_4s = model.set_variable(var_type='_x',  var_name='h_4s', shape=(1,1))  # Tank 4 fill level [m]

    # Input struct (optimization variables):
    v_1 = model.set_variable(var_type='_u',  var_name='v_1') # Pump 1 input [V]
    v_2 = model.set_variable(var_type='_u',  var_name='v_2') # Pump 2 input [V]

    # Fixed parameters:
    gamma_1 = model.set_variable(var_type='_p', var_name='gamma_1') # Fraction Pump 1 flow bypassed into Tank 1 (otherwise into Tank 4) []
    gamma_2 = model.set_variable(var_type='_p', var_name='gamma_2') # Fraction Pump 2 flow bypassed into Tank 2 (otherwise into Tank 3) []

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    model.set_expression('goal', goal)
    model.set_expression(expr_name='track_error', expr = sum1((goal - vertcat(h_1s, h_2s))**2))

    # Differential equations (includes rescaling)
    model.set_rhs('h_1s', -a_1*1e-4/A_1*(2*g*h_1s)**(1/2) + a_3*1e-4/A_1*(2*g*h_3s)**(1/2) + gamma_1*k_1*v_1/A_1)
    model.set_rhs('h_2s', -a_2*1e-4/A_2*(2*g*h_2s)**(1/2) + a_4*1e-4/A_2*(2*g*h_4s)**(1/2) + gamma_2*k_2*v_2/A_2)
    model.set_rhs('h_3s', -a_3*1e-4/A_3*(2*g*h_3s)**(1/2) + (1-gamma_2)*k_2*v_2/A_3)
    model.set_rhs('h_4s', -a_4*1e-4/A_4*(2*g*h_4s)**(1/2) + (1-gamma_1)*k_1*v_1/A_4)

    # Build the model
    model.setup()

    return model