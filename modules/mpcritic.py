import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c

import do_mpc
import casadi as ca
from copy import copy

from neuromancer.dataset import DictDataset
from neuromancer.system import Node, System
from neuromancer.constraint import variable, Objective
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem

kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}

class InputConcat(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, u):
        """ f(x,u) -> f(z) """
        # L4CasADi models must have 1 input (z), but Neuromancer needs 2 (x,u)
        z = torch.concat((x,u), dim=-1)
        return self.module(z)

class MPCritic(nn.Module):
    def __init__(self, model, dynamics, mpc_lterm, mpc_mterm, dpc, unc_p=None, mpc_settings=None):
        super().__init__()
        self.model = model
        self.dynamics = dynamics
        self.mpc_model = dynamics.dx.module # assumes dynamics.dx is InputConcat object
        self.mpc_lterm = mpc_lterm
        self.mpc_mterm = mpc_mterm
        self.dpc = dpc
        self.unc_p = unc_p
        self.batched_fwd_s = torch.vmap(self.forward_critic_s)
        self.batched_fwd_sa = torch.vmap(self.forward_critic_sa)

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

        self.mpc_settings = {'n_horizon': 5,
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
        
        self.node_dict = None
        self.problem = None

    def forward(self, s, a=None):
        """ batched critic operation """
        if a != None:
            return self.batched_fwd_sa(s, a)
        else:
            return self.batched_fwd_s(s)
    
    def forward_critic_s(self, s):
        """ single-batched critic(s,\mu(s)) operation """
        x = self.node_dict['x_shift']({'x_next' : s})
        u_pi = self.node_dict['policy']({'x_next' : s})
        input_dict = DictDataset({'x' : x['x'].view(1,1,-1),
                                  'u' : u_pi['u'].view(1,1,-1)})
        input_dict.datadict['name'] = 'eval'

        output_dict = self.problem(input_dict.datadict)
        # print(f"X: {output_dict['eval_x']}\n U: {output_dict['eval_u']}")
        return -output_dict['eval_loss'].view(1)
    
    def forward_critic_sa(self, s, a):
        """ single-batched critic(s,a) operation """
        input_dict = DictDataset({'x' : s.view(1,1,-1),
                                  'u' : a.view(1,1,-1)})
        input_dict.datadict['name'] = 'eval'

        output_dict = self.problem(input_dict.datadict)
        # print(f"X: {output_dict['eval_x']}\n U: {output_dict['eval_u']}")
        return -output_dict['eval_loss'].view(1)
    
    def forward_mpc(self, x0):
        """ single-batched MPC operation """
        return self.mpc.make_step(x0)
    
    def l4c_update(self):
        """ update all L4CasADi objects (do prior to recalling setup_mpc) """
        self.l4c_lterm = l4c.L4CasADi(self.mpc_lterm, **self.l4c_kwargs)
        self.l4c_mterm = l4c.L4CasADi(self.mpc_mterm, **self.l4c_kwargs)
        self.l4c_model = l4c.L4CasADi(self.mpc_model, **self.l4c_kwargs)
    
    def setup_mpc(self):
        """ setup MPC problem """
        mpc = do_mpc.controller.MPC(self.model)
        mpc.settings.__dict__.update(**self.mpc_settings)
        mpc.settings.supress_ipopt_output() # please be quiet

        z = ca.transpose(ca.vertcat(self.model._x, self.model._u))
        lterm = self.l4c_lterm.forward(z)
        x = ca.transpose(self.model._x)
        mterm = self.l4c_mterm.forward(x)
        # forward to 'build' l4c_model, required before L4CasADi.update()
        self.l4c_model(z)

        mpc.set_objective(lterm=lterm, mterm=mterm)
        mpc.set_rterm(u=0.)

        mpc.set_uncertainty_values(**self.unc_p)

        mpc.setup()

        self.mpc = mpc

    def init_mpc(self, x0):
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

    def online_mpc_update(self, x0, full=True):
        """ update MPC online, copying old MPC data into new MPC """
        mpc_data, mpc_t0 = copy(self.mpc.data), copy(self.mpc._t0)
        if full == True:
            self.l4c_update() # update l4c_lterm, mterm, model
        self.setup_mpc() # update model/uncertain parameters
        self.init_mpc(x0)

        self.mpc.data, self.mpc._t0 = mpc_data, mpc_t0

    def get_node_dict(self, mode='critic'):
        """ construct critic/DPC system nodes """
        if mode == 'dpc':
            # only used training the dpc with Neuromancer's Trainer class at the moment
            # concat_dynamics = InputConcat(self.mpc_model)
            # dynamics_model = Node(concat_dynamics, ['x', 'u'], ['x'], name='dynamics_model')
            dynamics_model = Node(self.dynamics.dx, ['x', 'u'], ['x'], name='dynamics_model')
            policy = Node(self.dpc, ['x'], ['u'], name='policy')

            concat_cost = InputConcat(self.mpc_lterm)
            stage_cost = Node(concat_cost, ['x', 'u'], ['l'], name='stage_cost')
            terminal_cost = Node(self.mpc_mterm, ['x'], ['V'], name='terminal_cost')

            node_list = [policy, dynamics_model, stage_cost, terminal_cost]

        elif mode == 'critic':
            # concat_dynamics = InputConcat(self.mpc_model)
            # dynamics_model = Node(concat_dynamics, ['x', 'u'], ['x_next'], name='dynamics_model')
            dynamics_model = Node(self.dynamics.dx, ['x', 'u'], ['x_next'], name='dynamics_model')
            policy = Node(self.dpc, ['x_next'], ['u'], name='policy')
            x_shift = Node(lambda x: x, ['x_next'], ['x'], name='x_shift')

            concat_cost = InputConcat(self.mpc_lterm)
            stage_cost = Node(concat_cost, ['x', 'u'], ['l'], name='stage_cost')
            terminal_cost = Node(self.mpc_mterm, ['x'], ['V'], name='terminal_cost')

            node_list = [dynamics_model, policy, x_shift, stage_cost, terminal_cost]

        self.node_dict = {node.name:node for node in node_list}
        return self.node_dict

    def setup_problem(self, mode='critic'):
        """ setup critic/DPC problem """
        l = variable('l')
        V = variable('V')
        # Ignore last stage cost (k=mpc_settings['n_horizon']+1), only consider last temrinal cost (k=mpc_settings['n_horizon']+2).
        # Stage cost should be weighted (mpc_settings['n_horizon']+1) times more than terminal cost.
        # This is because Neuromancer computes mean(stage costs) rather than mean(sum of stage_costs over horizon) for this formulation.
        stage_loss = Objective(var=(self.mpc_settings['n_horizon']+1.)*l[:, :-1, :], name='stage_loss')
        terminal_loss = Objective(var=(V[:, [-1], :]), name='terminal_loss')
        dpc_loss = PenaltyLoss([stage_loss, terminal_loss], [])

        # mpc_settings['n_horizon']+1 redundant terminal cost evaluations
        # 1 redundant stage cost evaluations
        # but objective matches that of do-mpc, horizon considered :)
        self.node_dict = self.get_node_dict(mode)
        cl_system = System(list(self.node_dict.values()), nsteps=self.mpc_settings['n_horizon'] + 2)
        problem = Problem([cl_system], dpc_loss)

        self.problem = problem
        return problem

if __name__ == '__main__':
    import numpy as np
    import gymnasium as gym
    from gymnasium.spaces.box import Box
    from stable_baselines3.common.buffers import ReplayBuffer

    from mpcomponents import QuadraticStageCost, QuadraticTerminalCost, LinearDynamics, LinearPolicy
    from dynamics import Dynamics
    from templates import template_linear_model, LQREnv
    from utils import calc_K, calc_P

    np_kwargs = {'dtype' : np.float32}

    """ System information """

    b = 1
    n = 2
    m = n
    # x = torch.ones((b,n), **kwargs) * torch.arange(b).view(b,1)
    # x[:,0] += 1.

    model = template_linear_model(n, m)
    
    Q, R = np.diag(np.ones(n, **np_kwargs)), np.diag(np.ones(m, **np_kwargs))
    A = np.diag(np.ones(n, **np_kwargs))
    B = np.diag(np.ones(m, **np_kwargs))
    K = -0.5 * np.diag(np.ones(n, **np_kwargs))
    unc_p = {'A' : [A],
             'B' : [B]}
    
    """ RL preliminaries """
    gym.register(
        id="gymnasium_env/LQR-v0",
        entry_point=LQREnv,
    )

    env_kwargs = {k:v for k,v in zip(["n", "m", "Q", "R", "A", "B"], [n, m, Q, R, A, B])}
    env = gym.make_vec("gymnasium_env/LQR-v0", num_envs=b, **env_kwargs)

    rb = ReplayBuffer(
        buffer_size=100000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=kwargs['device'],
        handle_timeout_termination=False,
    )

    """ Agent information """
    mpc_lterm = QuadraticStageCost(n, m, Q, R)
    mpc_mterm = QuadraticTerminalCost(n, Q)
    mpc_model = LinearDynamics(n, m, A, B)
    dpc = LinearPolicy(n, m, K)

    # test default behavior with state-action space NN model
    concat_dynamics = InputConcat(mpc_model)
    dynamics = Dynamics(env, rb, dx=concat_dynamics)

    critic = MPCritic(model, dynamics, mpc_lterm, mpc_mterm, dpc, unc_p)
    critic.setup_problem(mode='critic')

    """ Different outputs for given action """
    obs, _ = env.reset()
    x = torch.from_numpy(obs)

    q_s = critic(s=x)
    u = torch.zeros((b,m), **kwargs) # dpc(x)
    q_sa = critic(s=x, a=u)
    print(f'Q(s) != Q(s,a=0): {not torch.allclose(q_s, q_sa)}')

    """ Same outputs when action given/not given """
    q_s = critic(s=x)
    q_sa = critic(s=x, a=dpc(x))
    print(f'Q(s) == Q(s,a=\mu(s)): {torch.allclose(q_s, q_sa)}')

    """ Same outputs for optimal critic Q^* and value function V^* """
    K = calc_K(A, B, Q, R).astype(np_kwargs['dtype'])
    P = calc_P(A, B, Q, R).astype(np_kwargs['dtype'])
    mpc_mterm = QuadraticTerminalCost(n, P)
    dpc = LinearPolicy(n, m, K)
    critic = MPCritic(model, dynamics, mpc_lterm, mpc_mterm, dpc, unc_p)
    critic.setup_problem(mode='critic')

    q_s = critic(s=x)
    q_sa = critic(s=x, a=dpc(x))
    P = torch.from_numpy(P)
    V_s = -(x @ P * x).sum(axis=1, keepdims=True)
    print(f'Q^*(s) == Q^*(s,a=K^*(s))): {torch.allclose(q_s, q_sa)}')
    print(f'Q^*(s) == V^*(s)): {torch.allclose(q_s, V_s)}')