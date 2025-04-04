#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

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

def template_mpc(model, goal = 0.6, mpc_mode = "baseline", n_horizon = 20, silence_solver = True, solver="pardiso",
                RL_env=False, uncertain_params = "include_truth"):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    # Set settings of MPC:
    mpc.settings.n_robust = 1 # set to 0 if you just need a quick proof of concept
    mpc.settings.open_loop = False
    mpc.settings.t_step = 0.005
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

    mpc.bounds['lower', '_u', 'F'] = .5
    mpc.bounds['lower', '_u', 'Q_dot'] = -8.500

    mpc.bounds['upper', '_u', 'F'] = 10
    mpc.bounds['upper', '_u', 'Q_dot'] = 0.0

    if RL_env == True:
        # Any other requirements when specifying parameters of RL environment
        # (soft constraints for mpc, but temporarily assigned as a bound for creating the RL env)
        mpc.bounds['upper', '_x', 'T_R'] = 140

    if uncertain_params == "include_truth": # the default setting from do-mpc's examples
        alpha_var, beta_var = [np.array([1.0, 1.05, 0.95]), np.array([1.0, 1.1, 0.9])]
    if uncertain_params == "nominal": 
        alpha_var, beta_var = [np.array([1.0]), np.array([1.0])]

    mpc.set_uncertainty_values(alpha = alpha_var, beta = beta_var)

    if mpc_mode == "baseline" or mpc_mode == "nominal":
        mpc.settings.n_horizon = n_horizon # 20 is the original value
        if mpc_mode == "nominal":
            mpc.settings.n_robust = 0

        mpc.bounds['lower', '_x', 'C_a'] = 0.1
        mpc.bounds['lower', '_x', 'C_b'] = 0.1
        mpc.bounds['lower', '_x', 'T_R'] = 50
        mpc.bounds['lower', '_x', 'T_K'] = 50

        mpc.bounds['upper', '_x', 'C_a'] = 2
        mpc.bounds['upper', '_x', 'C_b'] = 2
        mpc.bounds['upper', '_x', 'T_K'] = 140

        mpc.scaling['_x', 'T_R'] = 100
        mpc.scaling['_x', 'T_K'] = 100
        mpc.scaling['_u', 'Q_dot'] = 2 # these are rescaled to account for the change of unit in template_model.py
        mpc.scaling['_u', 'F'] = 10 # these are rescaled to account for the change of unit

        lterm = (model.x['C_b'] - goal)**2
        mterm = (model.x['C_b'] - goal)**2
        mpc.set_objective(mterm=mterm, lterm=lterm)

        mpc.set_rterm(F = 1e-1, Q_dot = 1e-3)

        # Instead of having a regular bound on T_R:
        # We can also have soft constraints as part of the set_nl_cons method:
        mpc.set_nl_cons('T_R', model.x['T_R'], ub=140, soft_constraint=True, penalty_term_cons=1e2)

        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_now):
            tvp_template['_tvp',:, 'goal'] = goal
            return tvp_template
        mpc.set_tvp_fun(tvp_fun)
        
        mpc.setup()

    return mpc