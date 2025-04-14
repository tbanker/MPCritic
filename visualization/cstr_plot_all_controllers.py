import numpy as np
import torch
from casadi import *
import do_mpc
import copy
import gymnasium as gym

import sys
sys.path.append('')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.legend_handler import HandlerBase

from modules.mpcomponents import GoalMap

plt.style.use('visualization/ieee.mplstyle')
plt.style.use('tableau-colorblind10')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# to change the color order...
# colors=['#ABABAB','#006BA4','#FF800E'] # mpc, mpcritic (unconstrained), mpcritic (constrained)

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
constraint_color = 'gainsboro'
alpha = 0.55

def get_trajectories(controllers:dict, simulator, estimator, state_init=np.array([0.8, 0.4, 134.14, 130.0]), num_steps=100):

    simulator_data = {}

    for key, controller in controllers.items():

        x0 = state_init

        if 'rl' not in key: ## assume dictionary only contains RL policies or MPCs
            controller.reset_history()
            controller.x0 = x0
            controller.set_initial_guess()

        simulator.reset_history()
        estimator.reset_history()
        
        simulator.x0 = x0
        estimator.x0 = x0

        for _ in range(num_steps):
            if 'rl' in key:
                x = torch.tensor([np.transpose(x0)], dtype=torch.float32)
                u0= controller.get_deterministic_action(x)
                u0 = u0.detach().cpu().numpy()
                u0 = np.reshape(u0, (2, 1))
            else:
                u0 = controller.make_step(x0)

            x0 = simulator.make_step(u0)
            x0 = estimator.make_step(x0)
            # do_mpc.tools.printProgressBar(k, num_steps-1, prefix='Closed-loop simulation:', length=50)

        simulator_data[key] = copy.deepcopy(simulator.data)

    return simulator_data


def plot_trajectories(sim_data:dict, show_actions=False, labels=["RL", "MPC"], low=[0.1, 0.1, 50.0, 50.0] , high=[2.0, 2.0, 140.0, 140.0], num_steps=50, path=''):

    plt.rc('lines', linewidth=2.5)

    # Initialize graphic:
    if show_actions:
        num_plots = 6
    else:
        num_plots = 4

    fig, ax = plt.subplots(num_plots, sharex=True, layout='constrained')
    plt.xlim((0.0, 0.005*num_steps))

    class AnyObjectHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,
                        x0, y0, width, height, fontsize, trans):
            l1 = plt.Line2D([x0,y0+width], [0.75*height,0.75*height], color = colors[1])
            l2 = plt.Line2D([x0,y0+width], [0.20*height,0.20*height], color = colors[2])
            return [l1, l2]

    # Configure plot:
    lines = []
    for key, data in sim_data.items():
        graphics = do_mpc.graphics.Graphics(data)
        # graphics.clear()
        
        graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
        graphics.add_line(var_type='_x', var_name='C_b', axis=ax[1])
        graphics.add_line(var_type='_aux', var_name='goal', axis=ax[1], linestyle='dotted', color='grey')
        graphics.add_line(var_type='_x', var_name='T_R', axis=ax[2])
        graphics.add_line(var_type='_x', var_name='T_K', axis=ax[3])
    #     graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
    #     graphics.add_line(var_type='_aux', var_name='track_error', axis=ax[2])
        if show_actions:
            graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[4])
            graphics.add_line(var_type='_u', var_name='F', axis=ax[5])
        ax[0].set_ylabel(r'$c_A$')
        ax[1].set_ylabel(r'$c_B$')
        ax[2].set_ylabel(r'$T_R$')
        ax[3].set_ylabel(r'$T_K$')

        lines += graphics.result_lines['_x', 'C_a']
        
        if show_actions:
            ax[4].set_ylabel(r'$\dot{Q}$ [mW]')
            ax[5].set_ylabel('Flow [dal/h]')
            ax[5].set_xlabel('Time [h]')
        else:
            ax[3].set_xlabel('Time [h]')

        fig.align_ylabels()
        graphics.plot_results()

    ax[0].legend([lines[0], object], labels, handler_map={object: AnyObjectHandler()}, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.65))

    ax[0].annotate("unconstrained", (0.03, 2.55), (0.10, 2.50), arrowprops=dict(arrowstyle='->'), fontsize=10)

    ymin, ymax = ax[0].get_ylim()
    ax[0].fill_between(np.array([0.0, 0.005*num_steps]), np.max([low[0],ymin]), np.min([high[0], ymax]), color=constraint_color, alpha=alpha)
    ymin, ymax = ax[1].get_ylim()
    ax[1].fill_between(np.array([0.0, 0.005*num_steps]), np.max([low[1],ymin]), np.min([high[1], ymax]), color=constraint_color, alpha=alpha)
    ymin, ymax = ax[2].get_ylim()
    ax[2].fill_between(np.array([0.0, 0.005*num_steps]), np.max([low[2],ymin]), np.min([high[2], ymax]), color=constraint_color, alpha=alpha)
    ymin, ymax = ax[3].get_ylim()
    ax[3].fill_between(np.array([0.0, 0.005*num_steps]), np.max([low[3],ymin]), np.min([high[3], ymax]), color=constraint_color, alpha=alpha)

    fig.savefig("plotall")
    fig.savefig("plotall.pdf")

    return fig

def make_env(env_id, seed, idx, capture_video, run_name, path, goal_map=None):
    if "cstr" in env_id:
        from envs.CSTR.template_model import template_model
        from envs.CSTR.template_mpc import template_mpc
        from envs.CSTR.template_simulator import template_simulator
        from envs.DoMPCEnv import DoMPCEnv
        from envs.CSTR.cstr_plot import episode_plot

        gym.register(
        id=env_id,
        entry_point=DoMPCEnv,
            )  

        model = template_model()
        max_x = np.array([2.0,2.0,140.0,140.0]).flatten()
        min_x = np.array([0.1,0.1,50.0,50.0]).flatten() # writing like this to emphasize do-mpc sizing convention
        max_u = np.array([10.0,0.0]).flatten()
        min_u = np.array([0.5,-8.50]).flatten()
        bounds = {'x_low' : min_x, 'x_high' : max_x, 'u_low' : min_u, 'u_high' : max_u}
        goal_map = goal_map
        num_steps = 50
        goal_tol = 0.05
        kwargs = {'disable_env_checker': True, 'template_simulator': template_simulator, 'model': model, 'goal_tol': goal_tol,
                  'goal_map': goal_map, 'num_steps': num_steps, 'episode_plot': episode_plot, 'smooth_reward': True,
                 'bounds': bounds, 'same_state': None, 'path': path}

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

if __name__ == '__main__':

    from modules.networks import SACActor, Mu
    from envs.CSTR.template_model import template_model
    from envs.CSTR.template_mpc import template_mpc
    from envs.CSTR.template_simulator import template_simulator

    goal = 0.60 # don't change
    n_horizon = 5 # for nstep RL+MPC
    num_steps = 80

    # First run `sac_continuous_action.py` and swap out the experiment name below
    mpcritic_constrained_path = "runs/cstr-v0__sac_continuous_action__0__1744669827/" # H=5, scale=100

    goal_map_env = GoalMap(idx=[1], goal=goal)
    envs = gym.vector.SyncVectorEnv(
        [make_env("cstr-v0", 1, 0, False, "", mpcritic_constrained_path, goal_map=goal_map_env) for i in range(1)]
    )

    state_init = np.array([0.8, 0.3, 100.14, 100.0]) # MPC gets stuck -- run with eval_uncertain_env=True to get situations where both RL and MPC fail but together they succeed

    ## Get actor network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_constrained = SACActor(envs).to(device)

    actor_constrained.load_state_dict(torch.load(mpcritic_constrained_path+"/actor.pth", weights_only=True))
    actor_constrained.eval()

    ## Optional: setup MPC
    model = template_model()
    # mpc = template_mpc(model, goal=goal, silence_solver=True)
    estimator = do_mpc.estimator.StateFeedback(model)
    simulator = template_simulator(model, uncertain_params="nominal", goal=goal)

    controllers = {        
                       "rl": actor_constrained  
}

    sim_data = get_trajectories(controllers, simulator, estimator, state_init=state_init, num_steps=num_steps)

    labels=[r"\texttt{MPCritic}"]
    plot_trajectories(sim_data, num_steps=num_steps, labels=labels)