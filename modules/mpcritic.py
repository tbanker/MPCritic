import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c

import do_mpc
import casadi as ca
from copy import copy

kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}

class MPCritic(nn.Module):
    def __init__(self, model, mpc_model, mpc_lterm, mpc_mterm, dpc, unc_p=None, mpc_settings=None):
        super().__init__()
        self.model = model
        self.mpc_model = mpc_model
        self.mpc_lterm = mpc_lterm
        self.mpc_mterm = mpc_mterm
        self.dpc = dpc
        self.unc_p = unc_p

        self.l4c_kwargs = {'device' : 'cpu',
                           'batched' : True,
                           'mutable' : True,
                           'generate_jac' : True,
                           'generate_jac_jac' : True,
                           'generate_jac_adj1' : True,
                           'generate_adj1' : False,} # LQR fails w/ this
        
        self.l4c_lterm = l4c.L4CasADi(self.mpc_lterm, **self.l4c_kwargs)
        self.l4c_mterm = l4c.L4CasADi(self.mpc_mterm, **self.l4c_kwargs)
        self.l4c_model = l4c.L4CasADi(self.mpc_model, **self.l4c_kwargs)

        self.mpc_settings = {'n_horizon': 10,
                             'n_robust': 0,
                             'open_loop': False,
                             't_step': 1.0,
                            #  'use_terminal_bounds' : False,
                            #  'state_discretization' : 'collocation',
                            #  'collocation_type': 'radau',
                            #  'collocation_deg': 2,
                            #  'collocation_ni': 2,
                            #  'nl_cons_check_colloc_points' : False,
                            #  'nl_cons_single_slack' : False,
                            #  'cons_check_colloc_points' : True,
                             'store_full_solution': True,
                            #  'store_lagr_multiplier' : True,
                            #  'store_solver_stats' : []
                             'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}, #LQR fails w/ MA27
                             } if mpc_settings == None else mpc_settings

    def forward(self, s, a):
        """ temporaily, LQR critic """
        H = self.mpc_settings['n_horizon']
        b = s.shape[0]
        
        x, u = s, a

        q = torch.zeros((b,1), **kwargs)
        for k in range(H):
            z = torch.concat((x,u), dim=-1)
            q += self.mpc_lterm(z)
            x = self.mpc_model(z)
            u = self.dpc(x)
        q += self.mpc_mterm(x)

        return -q # negative to match mpc minimization
    
    def forward_mpc(self, x0):
        return self.mpc.make_step(x0)
    
    def template_mpc(self):
        mpc = do_mpc.controller.MPC(self.model)
        mpc.settings.__dict__.update(**self.mpc_settings)
        mpc.settings.supress_ipopt_output() # please be quiet

        z = ca.transpose(ca.vertcat(self.model._x, self.model._u))
        lterm = self.l4c_lterm.forward(z)
        x = ca.transpose(self.model._x)
        mterm = self.l4c_mterm.forward(x)
        self.l4c_model(z)
        # forward to "build" l4c_model, required before L4CasADi.update()
        # l4c_model unused atm in the MPC tho

        mpc.set_objective(lterm=lterm, mterm=mterm)
        mpc.set_rterm(u=0.)

        # Original stategy at updating MPC model parameters (worked okay)
        # model_params = {}
        # for name, param in self.mpc_model.named_parameters():
        #     model_params[name] = [param.numpy()] # 1 uncertainty scenario considered
        # mpc.set_uncertainty_values(**model_params)

        # Alternative stategy at updating MPC model parameters;
        mpc.set_uncertainty_values(**self.unc_p)

        mpc.setup()

        self.mpc = mpc

    def init_mpc(self, x0):
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    def l4c_update(self):
        """ update all L4CasADi objects (do prior to recalling template_mpc) """

        # why does this not work :(
        # self.l4c_lterm.update()
        # self.l4c_mterm.update()
        # self.l4c_model.update()

        self.l4c_lterm = l4c.L4CasADi(self.mpc_lterm, **self.l4c_kwargs)
        self.l4c_mterm = l4c.L4CasADi(self.mpc_mterm, **self.l4c_kwargs)
        self.l4c_model = l4c.L4CasADi(self.mpc_model, **self.l4c_kwargs)

    def online_update(self, x0, full=True):
        """ update MPC online, copying old MPC data into new MPC """
        mpc_data, mpc_t0 = copy(self.mpc.data), copy(self.mpc._t0)
        if full == True:
            self.l4c_update() # update l4c_lterm, mterm, model
        self.template_mpc() # update model/uncertain parameters
        self.init_mpc(x0)

        self.mpc.data, self.mpc._t0 = mpc_data, mpc_t0
    
if __name__ == '__main__':
    import numpy as np

    from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, LinearDynamics, LinearPolicy
    from templates import template_linear_model

    np_kwargs = {'dtype' : np.float32}

    b, n, m = 3, 4, 2
    x, u = torch.ones((b,n), **kwargs), torch.ones((b,m), **kwargs)
    x[:,0], u[:,0] = 2., 2.
    z = torch.concat((x,u), dim=-1)

    model = template_linear_model(n, m)
    
    Q, R = np.ones((n,n), **np_kwargs), np.ones((m,m), **np_kwargs)
    A, B = np.ones((n,n), **np_kwargs), np.ones((n,m), **np_kwargs)
    K = np.ones((m,n), **np_kwargs)
    unc_p = {'A' : [A],
             'B' : [B]}

    mpc_lterm = QuadraticStageCost(n, m, Q, R)
    mpc_mterm = QuadraticTerminalCost(n, Q)
    mpc_model = LinearDynamics(n, m, A, B)
    dpc = LinearPolicy(n, m, K)

    critic = MPCritic(model, mpc_model, mpc_lterm, mpc_mterm, dpc, unc_p)
    critic.template_mpc()

    q = critic(s=x, a=u)
    print(q)

    print("Success!")