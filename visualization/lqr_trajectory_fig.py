import os
import sys
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

import gymnasium as gym
from do_mpc.graphics import Graphics
from neuromancer.modules import blocks

sys.path.append('')
from algos.ddpg_continuous_action import Args, Actor
from modules.mpcomponents import QuadraticStageCost, QuadraticTerminalCost, PDQuadraticTerminalCost, LinearDynamics, LinearPolicy, GoalMap
from modules.mpcritic import MPCritic, InputConcat
from modules.dynamics import Dynamics
from modules.dpcontrol import DPControl

plt.style.use('visualization/ieee.mplstyle')
plt.style.use('tableau-colorblind10')

prim_ln_kwargs = {
    'ls' : '-',
    'lw' : 2.0,
    'marker' : 'o',
    'ms' : 5.0,
    'mec' : 'none',
    'mew' : 0.5,
    'mfc' : 'none',
}

sec_ln_kwargs = {
    'ls' : '-',
    'lw' : 1.0,
    'marker' : 'o',
    'ms' : 5.0,
    'mec' : 'none',
    'mew' : 0.5,
    'mfc' : 'none',
}

def make_env(env_id, seed, idx, capture_video, run_name, path, goal_map, same_state, num_steps):
    if "lqr" in env_id:
        from envs.LQR.template_model import template_model
        from envs.LQR.template_mpc import template_mpc
        from envs.LQR.template_simulator import template_simulator
        from envs.DoMPCEnv import DoMPCEnv

        gym.register(
        id=env_id,
        entry_point=DoMPCEnv,
            )  

        model = template_model(n=args.n, m=args.n)
        max_x = np.ones(args.n).flatten()
        min_x = -np.ones(args.n).flatten() # writing like this to emphasize do-mpc sizing convention
        max_u = np.ones(args.n).flatten()
        min_u = -np.ones(args.n).flatten()
        bounds = {'x_low' : min_x, 'x_high' : max_x, 'u_low' : min_u, 'u_high' : max_u}
        goal_map = goal_map
        num_steps = num_steps
        kwargs = {'disable_env_checker': True, 'template_simulator': template_simulator, 'model': model,
                'num_steps': num_steps, 'bounds': bounds, 'same_state': same_state,
                'goal_map': goal_map, 'smooth_reward': False, 'sa_reward': True,
                'path': path}

    else:
        kwargs = {}
    
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def make_envs(args, same_state, num_steps):
    goal_map = GoalMap()
    envs_mpc = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, '', None, goal_map, same_state, num_steps+1)]) # +1 prevent autoreset before collecting data
    envs_rl = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, '', None, goal_map, same_state, num_steps+1)])
    envs_mu = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, '', None, goal_map, same_state, num_steps+1)])

    return {'mpc': envs_mpc, 'rl': envs_rl, 'mu': envs_mu}

def load_controllers(run_names:dict, envs, global_step=None):
    goal_map = GoalMap()
    V = PDQuadraticTerminalCost(np.array(envs.single_observation_space.shape).prod())
    mu = blocks.MLP_bounds(insize=np.array(envs.single_observation_space.shape).prod(), outsize=np.array(envs.single_action_space.shape).prod(), bias=True, linear_map=torch.nn.Linear, nonlin=torch.nn.ReLU, hsizes=[100 for h in range(2)], min=torch.from_numpy(envs.action_space.low), max=torch.from_numpy(envs.action_space.high))
    dpcontrol = DPControl(envs, H=10, mu=mu, linear_dynamics=True, V=V, rb=None, goal_map=goal_map, lr=args.learning_rate, xlim=np.array([envs.observation_space.low,envs.observation_space.high]), ulim=np.concatenate([envs.action_space.low,envs.action_space.high], axis=0)).to('cpu')
    mpcritic = MPCritic(dpcontrol).to('cpu')
    # qf1.setup_mpc()
    # qf1.requires_grad_(True)
    mpcritic_state_dict = torch.load(f"runs/{run_names['mpc']}/mpcritic.pt") if global_step is None else torch.load(f"runs/{run_names['mpc']}/mpcritic{global_step}.pt")
    mpcritic.load_state_dict(mpcritic_state_dict)
    mpcritic.l4c_update()
    mpcritic.setup_mpc()

    actor = Actor(envs).to('cpu')
    actor_state_dict = torch.load(f"runs/{run_names['rl']}/actor.pt") if global_step is None else torch.load(f"runs/{run_names['rl']}/actor{global_step}.pt")
    actor.load_state_dict(actor_state_dict)

    return {'mpc': mpcritic, 'rl': actor, 'mu': mpcritic.dpcontrol.mu}

def get_traj_datas(controllers:dict, envs:dict, num_steps, args):
    
    datas = {k:None for k,v in controllers.items()}
    episodic_rewards = {k:None for k,v in controllers.items()}
    
    with torch.no_grad():
        for key, controller in controllers.items():
            episodic_reward = 0

            obs, _ = envs[key].reset(seed=args.seed)
            envs[key].envs[0].simulator.reset_history() # data.init_storage()
            for step in range(num_steps):
                if key == 'mpc':
                    actions = controller._rl_action(controller.forward_mpc(controller._mpc_state(obs)))
                elif key == 'mu':
                    actions = controller(torch.Tensor(obs).to('cpu'))
                elif key == 'rl':
                    actions = controller(torch.Tensor(obs).to('cpu'))
                actions = actions.cpu().numpy()#.clip(envs.single_action_space.low, envs.single_action_space.high)

                next_obs, rewards, terminations, truncations, infos = envs[key].step(actions)
                obs = next_obs
                episodic_reward+=rewards

            datas[key] = copy.deepcopy(envs[key].envs[0].simulator.data)
            episodic_rewards[key] = episodic_reward

    return datas, episodic_rewards

def get_mu_preds(controller, x0):
    f = controller.dpcontrol.dynamics.dx
    mu = controller.dpcontrol.mu

    xs = [x0]
    us = []
    for i in range(controller.dpcontrol.H):
        us.append(mu(torch.Tensor(xs[-1])).detach().numpy())
        xs.append(f(torch.Tensor(xs[-1]),torch.Tensor(us[-1])).detach().numpy())

    return {
        '_x' : np.concatenate(xs, axis=0),
        '_u' : np.concatenate(us, axis=0),
    }

def get_mpc_preds(controller, x0):
    # controller.setup_mpc()
    x0 = controller._mpc_state(x0)
    controller.mpc.x0 = x0
    controller.forward_mpc(x0)

    return {
        '_x' : copy.deepcopy(controller.mpc.data.prediction(('_x', 'x'), -1).T.squeeze()),
        '_u' : copy.deepcopy(controller.mpc.data.prediction(('_u', 'u'), -1).T.squeeze())
    }
    # return copy.deepcopy(controller.mpc.data)

def plot_traj(datas:dict, labels, colors, dim=0):

    fig, axes = plt.subplots(3, 1, sharex=True, layout='constrained', figsize=(3.3,4), height_ratios=[1,1,2])

    max_s_norm, max_a_norm = 0, 0
    for key, data in datas.items():

        s_norm = np.linalg.norm(data['_x'], ord=np.inf, axis=1)
        axes[0].plot(data['_time'], s_norm, label=labels[key], color=colors[key], **prim_ln_kwargs)

        a_norm = np.linalg.norm(data['_u'], ord=np.inf, axis=1)
        axes[1].plot(data['_time'], a_norm, label=None, color=colors[key], **prim_ln_kwargs)

        max_s_norm = np.max(s_norm) if np.max(s_norm) > max_s_norm else max_s_norm
        max_a_norm = np.max(s_norm) if np.max(s_norm) > max_a_norm else max_a_norm

    s_lim = 0.05+max_s_norm if max_s_norm > 1 else 1.05
    a_lim = 0.05+max_a_norm if max_a_norm > 1 else 1.05
    axes[0].fill_between(datas['mpc']['_time'].flatten(), np.ones_like(datas['mpc']['_time']).flatten(), s_lim*np.ones_like(datas['mpc']['_time']).flatten(), color='r', alpha=0.1)
    axes[1].fill_between(datas['mpc']['_time'].flatten(), np.ones_like(datas['mpc']['_time']).flatten(), a_lim*np.ones_like(datas['mpc']['_time']).flatten(), color='r', alpha=0.1)
    axes[2].fill_between(datas['mpc']['_time'].flatten(), np.ones_like(datas['mpc']['_time']).flatten(), s_lim*np.ones_like(datas['mpc']['_time']).flatten(), color='r', alpha=0.1)
    axes[2].fill_between(datas['mpc']['_time'].flatten(), -np.ones_like(datas['mpc']['_time']).flatten(), -s_lim*np.ones_like(datas['mpc']['_time']).flatten(), color='r', alpha=0.1)
    # axes[0].plot(datas['mpc']['_time'], np.ones_like(datas['mpc']['_time']), ls='--', color='black')
    # axes[1].plot(datas['mpc']['_time'], np.ones_like(datas['mpc']['_time']), ls='--', color='black')
    # axes[2].plot(datas['mpc']['_time'], -np.ones_like(datas['mpc']['_time']), ls='--', color='black')
    # axes[2].plot(datas['mpc']['_time'], np.ones_like(datas['mpc']['_time']), ls='--', color='black')

    # switch to mpc traj
    axes[2].plot(data['_time'], datas['rl']['_x'][:,dim], color=colors['rl'], **prim_ln_kwargs)
    for t in datas['mpc']['_time'][::3].flatten().astype(np.int64):
        obs = datas['rl']['_x'][[t]]
        mpc_data = get_mpc_preds(controllers['mpc'], obs)
        mu_data = get_mu_preds(controllers['mpc'], obs)

        # predictions
        # axes[2].plot(np.arange(t,t+len(mpc_data['_x'])), mpc_data['_x'][:,dim], ls=(0,(1,1)), color=colors['mpc'])
        # axes[2].plot(np.arange(t,t+len(mu_data['_x'])), mu_data['_x'][:,dim], ls=(0,(1,1)), color=colors['mu'])
        axes[2].plot(np.arange(t,t+len(mpc_data['_x'])), mpc_data['_x'][:,dim], color=colors['mpc'], alpha=0.8, **sec_ln_kwargs)
        axes[2].plot(np.arange(t,t+len(mu_data['_x'])), mu_data['_x'][:,dim], color=colors['mu'], alpha=0.8, **sec_ln_kwargs)
        # start and end points
        axes[2].plot(t, mpc_data['_x'][0,dim], marker='o', ms=2, mfc='none', ls='none', color='black')
        axes[2].plot(t+len(mpc_data['_x'])-1, mpc_data['_x'][-1,dim], marker='o', ms=2, mfc='none', ls='none', color=colors['mpc'])
        axes[2].plot(t+len(mu_data['_x'])-1, mu_data['_x'][-1,dim], marker='o', ms=2, mfc='none', ls='none', color=colors['mu'])
        
    

    axes[0].set_xlim(datas['mpc']['_time'][0], datas['mpc']['_time'][-1])
    axes[0].set_ylim(0,s_lim)
    axes[0].set_ylabel(r'$||s||_{\infty}$')

    axes[1].set_ylim(0,a_lim)
    axes[1].set_ylabel(r'$||a||_{\infty}$')

    axes[2].set_ylim(-s_lim,s_lim)
    axes[2].set_ylabel(f'$s_{dim}$')

    axes[-1].set_xlabel(r'$t$')

    axes[0].legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5,1.5))

    return fig, axes


if __name__ == '__main__':
    
    args = Args()
    args.seed = 2
    date = '2025-03-24'
    run_time_mpcritic = 1742940149 # (50001 steps)
    run_time_rl = 1742950624 # (50001 steps)
    global_step = 30000
    run_names = {
        'mpc' : f"{args.env_id}__{args.exp_name}__{args.seed}__{run_time_mpcritic}",
        'rl'       : f"{args.env_id}__{args.exp_name}__{args.seed}__{run_time_rl}",
    }


    for _ in range(200):
        same_state = np.random.uniform(-1,1,4) # 0.9*np.ones(4)
        num_steps = 50
        
        envs = make_envs(args, same_state, num_steps)
        controllers = load_controllers(run_names, envs['mpc'], global_step=global_step)

        datas, episodic_rewards = get_traj_datas(controllers, envs, num_steps, args)
        
        colors = {k:f'C{i}' for i, k in enumerate(controllers.keys())}
        labels = {'mpc':r'$\pi^{\text{MPC}}$', 'rl':r'$\pi^{\text{DNN}}$', 'mu':r'$\mu$'}
        fig, axes = plot_traj(datas, labels, colors, dim=0)

        save_dir = os.path.join(os.path.dirname(__file__), "LQR")
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{date}_LQR_traj_{np.round(same_state,3).tolist()}.png"))

        print(same_state)
        for label, episodic_reward in zip(labels, episodic_rewards.values()):
            print(f"{label}: R={episodic_reward}")