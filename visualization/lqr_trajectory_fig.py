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
# from algos.ddpg_continuous_action import Args, Actor
from algos.td3_continuous_action import Args, Actor
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
    x0 = controller._mpc_state(x0)
    controller.mpc.x0 = x0
    controller.forward_mpc(x0)

    return {
        '_x' : copy.deepcopy(controller.mpc.data.prediction(('_x', 'x'), -1).T.squeeze()),
        '_u' : copy.deepcopy(controller.mpc.data.prediction(('_u', 'u'), -1).T.squeeze())
    }

def plot_traj(datas:dict, labels, colors, dim=0):

    fig, axes = plt.subplots(2, 1, sharex=True, layout='constrained', figsize=(3.3,3.3), height_ratios=[2,1])

    max_s_norm, max_a_norm = 0, 0

    if type(datas) != list:
        for key, data in datas.items():

                s_norm = np.linalg.norm(data['_x'], ord=np.inf, axis=1)
                a_norm = np.linalg.norm(data['_u'], ord=np.inf, axis=1)
                max_s_norm = np.max(s_norm) if np.max(s_norm) > max_s_norm else max_s_norm
                max_a_norm = np.max(s_norm) if np.max(s_norm) > max_a_norm else max_a_norm

                axes[0].plot(data['_time'], s_norm, label=labels[key], color=colors[key], **prim_ln_kwargs)
                axes[1].plot(data['_time'], a_norm, label=None, color=colors[key], **prim_ln_kwargs)
    else:
        data = datas[0]['mpc']
        empty_plt_dict = {k:np.zeros_like(datas[0][k]['_time']) for k in datas[0].keys()}
        min_s_norm_plt = copy.deepcopy(empty_plt_dict)
        max_s_norm_plt = copy.deepcopy(empty_plt_dict)
        mean_s_norm_plt = copy.deepcopy(empty_plt_dict)
        min_a_norm_plt = copy.deepcopy(empty_plt_dict)
        max_a_norm_plt = copy.deepcopy(empty_plt_dict)
        mean_a_norm_plt = copy.deepcopy(empty_plt_dict)
        for key in empty_plt_dict.keys():
            for s in range(len(datas)):
                s_norm = np.linalg.norm(datas[s][key]['_x'], ord=np.inf, axis=1, keepdims=True)
                a_norm = np.linalg.norm(datas[s][key]['_u'], ord=np.inf, axis=1, keepdims=True)
                max_s_norm = np.max(s_norm) if np.max(s_norm) > max_s_norm else max_s_norm
                max_a_norm = np.max(a_norm) if np.max(a_norm) > max_a_norm else max_a_norm

                if s == 0:
                    min_s_norm_plt[key], max_s_norm_plt[key] = s_norm.copy(), s_norm.copy()
                    min_a_norm_plt[key], max_a_norm_plt[key] = a_norm.copy(), a_norm.copy()
                else:
                    min_s_norm_plt[key][s_norm < min_s_norm_plt[key]] = s_norm[s_norm < min_s_norm_plt[key]].copy()
                    max_s_norm_plt[key][s_norm > max_s_norm_plt[key]] = s_norm[s_norm > max_s_norm_plt[key]].copy()
                    min_a_norm_plt[key][a_norm < min_a_norm_plt[key]] = a_norm[a_norm < min_a_norm_plt[key]].copy()
                    max_a_norm_plt[key][a_norm > max_a_norm_plt[key]] = a_norm[a_norm > max_a_norm_plt[key]].copy()
                mean_s_norm_plt[key] += s_norm/len(datas)
                mean_a_norm_plt[key] += a_norm/len(datas)

            axes[0].fill_between(data['_time'].flatten(), min_s_norm_plt[key].flatten(), max_s_norm_plt[key].flatten(), color=colors[key], alpha=0.3, step="post")
            axes[1].fill_between(data['_time'].flatten(), min_a_norm_plt[key].flatten(), max_a_norm_plt[key].flatten(), color=colors[key], alpha=0.3, step="post")

        for s in range(len(datas)):
            for key, data in datas[s].items():

                label = labels[key] if s==0 else None
                s_norm = np.linalg.norm(data['_x'], ord=np.inf, axis=1)
                a_norm = np.linalg.norm(data['_u'], ord=np.inf, axis=1)

                axes[0].step(data['_time'], s_norm, label=label, color=colors[key], alpha=0.5, where='post', **sec_ln_kwargs)
                axes[1].step(data['_time'], a_norm, label=None, color=colors[key], alpha=0.5, where='post', **sec_ln_kwargs)


    s_lim = 0.05+max_s_norm if max_s_norm > 1 else 1.05
    a_lim = 0.05+max_a_norm if max_a_norm > 1 else 1.05
    axes[0].fill_between(data['_time'].flatten(), np.zeros_like(data['_time']).flatten(), np.ones_like(data['_time']).flatten(), color='gainsboro', alpha=0.6, lw=0., zorder=0)
    axes[1].fill_between(data['_time'].flatten(), np.zeros_like(data['_time']).flatten(), np.ones_like(data['_time']).flatten(), color='gainsboro', alpha=0.6, lw=0., zorder=0)

    # switch to mpc traj
    # axes[2].plot(data['_time'], datas['rl']['_x'][:,dim], color=colors['rl'], **prim_ln_kwargs)
    # for t in datas['mpc']['_time'][::3].flatten().astype(np.int64):
    #     obs = datas['rl']['_x'][[t]]
    #     mpc_data = get_mpc_preds(controllers['mpc'], obs)
    #     mu_data = get_mu_preds(controllers['mpc'], obs)

    #     # predictions
    #     axes[2].plot(np.arange(t,t+len(mpc_data['_x'])), mpc_data['_x'][:,dim], color=colors['mpc'], alpha=0.8, **sec_ln_kwargs)
    #     axes[2].plot(np.arange(t,t+len(mu_data['_x'])), mu_data['_x'][:,dim], color=colors['mu'], alpha=0.8, **sec_ln_kwargs)
    #     # start and end points
    #     axes[2].plot(t, mpc_data['_x'][0,dim], marker='o', ms=2, mfc='none', ls='none', color='black')
    #     axes[2].plot(t+len(mpc_data['_x'])-1, mpc_data['_x'][-1,dim], marker='o', ms=2, mfc='none', ls='none', color=colors['mpc'])
    #     axes[2].plot(t+len(mu_data['_x'])-1, mu_data['_x'][-1,dim], marker='o', ms=2, mfc='none', ls='none', color=colors['mu'])

    axes[0].set_xlim(data['_time'][0], data['_time'][-1])
    axes[0].set_ylim(0,s_lim)
    axes[0].set_yticks([0,1])
    axes[0].set_yticklabels(['0','1'])
    axes[0].set_ylabel(r'$\Vert s_{t} \Vert_{\infty}$')
    axes[1].set_ylim(0,a_lim)
    axes[1].set_yticks([0,1])
    axes[1].set_yticklabels(['0','1'])
    axes[1].set_ylabel(r'$\Vert a_{t} \Vert_{\infty}$')
    # axes[2].set_ylim(-s_lim,s_lim)
    # axes[2].set_ylabel(f'$s_{dim}$')
    axes[-1].set_xlabel(r'$t$')

    axes[0].legend(ncols=3, loc='upper center', columnspacing=1.)

    return fig, axes


if __name__ == '__main__':
    
    date = '2025-03-28'
    num_steps = 11

    """ Plot different seeds """
    global_step = None
    runs = {
        '1'  : [1743222882, 1743221115],
        '2'  : [1743232317, 1743221236],
        '3'  : [1743242050, 1743221356],
        '4'  : [1743251979, 1743221476],
        '5'  : [1743262112, 1743221599],
        '6'  : [1743272856, 1743221719],
        '7'  : [1743285714, 1743221841],
        '8'  : [1743296735, 1743221962],
        '9'  : [1743308000, 1743222086],
        '10' : [1743319284, 1743222208],
    }

    run_datas = []
    same_state = np.array([-0.885, 0.742, -0.659, -0.784])

    """ Plot different global steps """
    # seed = 1
    # global_steps = list(range(0,50001,10000))
    # mpc_run_timetime = 1743112746
    # rl_run_timetime = 1743111242
    # runs = {global_step:[mpc_run_timetime, rl_run_timetime] for global_step in global_steps}

    for seed, times in runs.items():
    # for global_step, times in runs.items():
        args = Args()
        args.seed = int(seed)
        run_names = {
            'mpc' : f"{args.env_id}__{args.exp_name}__{args.seed}__{times[0]}",
            'rl'       : f"{args.env_id}__{args.exp_name}__{args.seed}__{times[1]}",
        }
        
        envs = make_envs(args, same_state, num_steps)
        controllers = load_controllers(run_names, envs['mpc'], global_step=global_step)

        datas, episodic_rewards = get_traj_datas(controllers, envs, num_steps, args)
        run_datas.append(datas)
        
    colors = {'mpc': 'C0', 'rl': 'C1', 'mu': 'darkslategrey'}
    labels = {'mpc':r'$\pi^{\text{MPC}}$', 'rl':r'$\pi^{\text{DNN}}$', 'mu':r'$\mu$'}
    fig, axes = plot_traj(datas=run_datas, labels=labels, colors=colors, dim=0)

    save_dir = os.path.join(os.path.dirname(__file__), "LQR")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{date}_LQR_traj_{np.round(same_state,3).tolist()}.png"))

    print(same_state)
    for label, episodic_reward in zip(labels, episodic_rewards.values()):
        print(f"{label}: R={episodic_reward}")