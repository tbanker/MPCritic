"""
A Gym wrapper that serves as the bridge between SB3 and do-mpc
"""
import sys
sys.path.append('')

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

import do_mpc
from modules.mpcomponents import GoalMap


class DoMPCEnv(gym.Env):
    """
    Gym environment that uses do-mpc for carrying out simulations
    """
    # def __init__(self, simulator:do_mpc.simulator.Simulator, 
    def __init__(self, template_simulator:callable=None, 
                 model:do_mpc.model._model.Model=None,
                 bounds:dict=None, 
                 goal_map=None,
                    num_steps=200,
                    clip_reset=None,
                    same_state=None,
                    goal_tol=0.01,
                    uncertain_params = "nominal", # set the uncertainty keyword for template_simulator.py.
                    smooth_reward=True,
                    episode_plot=None,
                    sa_reward=False,
                    sa_reward_scale=10,
                    eval="",
                    path=''):        
        super().__init__()

        self.model = model
        self.template_simulator = template_simulator
        self.uncertain_params = uncertain_params
        self.num_steps = num_steps
        self.clip_reset = clip_reset
        self.same_state = same_state
        self.goal_map = goal_map if goal_map != None else GoalMap()
        self.episode = 0
        self.eval = eval
        self.path=path
        self.smooth_reward = smooth_reward
        self.sa_reward = sa_reward
        self.sa_reward_scale = sa_reward_scale
        self.bounds = bounds 
        self.n,_ = model._x.cat.shape
        self.m,_ = model._u.cat.shape
        if bounds is None:
            # n,_ = model._x.cat.shape
            # m,_ = model._u.cat.shape
            self.observation_space = gym.spaces.Box(
                low=np.full((self.n), -np.inf).astype(np.float32), high=np.full((self.n), np.inf).astype(np.float32),
                shape=(self.n,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=np.full((self.m), -np.inf).astype(np.float32), high=np.full((self.m), np.inf).astype(np.float32),
                shape=(self.m,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(low=bounds['u_low'], high=bounds['u_high'], dtype=np.float32)
            self.observation_space = spaces.Box(low=bounds['x_low'], high=bounds['x_high'], dtype=np.float32)
        
        self.episode_plot = episode_plot
        self.goal_tol = goal_tol

    def step(self, action):
        self.t += 1

        self.action = action
        if self.sa_reward:
            obs = self._simulator(action)
        else:
            self.state = self._simulator(action)
            
        # do_mpc.tools.printProgressBar(self.t, self.num_steps, prefix='Closed-loop simulation:', length=50)
    
        if self.sa_reward:
            info = self._get_info()
            self.state = obs
        else:
            obs = self.state
            info = self._get_info()
        reward, terminated, truncated = info["reward"], info["terminated"], info["TimeLimit.truncated"]

        return obs, reward, terminated, truncated, info
    
    def compute_reward(self, achieved_goal, info):

        # if type(info) is not dict:
        #     state = [x["full_state"] for x in info]
        #     is_feasible = [1*self.observation_space.contains(
        #         x) - 1 for x in state]
        # else:
        #     state = info["full_state"]
        #     is_feasible = 1*self.observation_space.contains(
        #         state) - 1

        axis = 0
        # if achieved_goal.ndim == 2:
        #     axis = 1

        # print(desired_goal, achieved_goal)
        # print(desired_goal - achieved_goal)
        is_target = np.abs(achieved_goal).sum() < self.goal_tol

        if self.smooth_reward:
            reward = np.exp(-0.5*(np.sum((achieved_goal)**2) / self.goal_tol)**2) - 1
            # reward = -0.5*np.abs(achieved_goal).sum()
        elif self.sa_reward:
            reward = -np.sum(self.state**2)-self.sa_reward_scale*np.sum(self.action**2)
        else:
            reward = 1*is_target
            # reward = 1*is_feasible + 1*is_target # can tune this, but we opt to use mpc for constraint handling

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode += 1

        # self.goal = 0.0 # the goal is embedded in goal_map for now)
        

        self.simulator = self._define_simulator()
        self.simulator.reset_history()

        x = self._reset_state()
        self.action = self.action_space.sample()
        self.simulator.x0 = x
        self.state = self._env_state(x)
        obs = self.state
        self.t = 0.0
        info = self._get_info()

        return obs, info
        
    def _reset_state(self, state=None):

        if self.same_state is not None: # similar to same_goal, dictates whether the system starts from the same state between episodes
            state = self.same_state
            return state
        elif self.clip_reset is not None:
            state = np.clip(self.observation_space.sample(), -self.clip_reset, self.clip_reset)
            return state
        else:
            # return self.observation_space.sample()
            return  self.np_random.uniform(low=self.bounds['x_low'], high=self.bounds['x_high'])

    def _get_info(self):
        reward = self.compute_reward(self.goal_map(self.state), {"full_state": self.state})

        distance = np.linalg.norm(self.goal_map(self.state)).item() # possible to set `terminated = distance < 0.0001`
        terminated = False
        # terminated = not self.observation_space.contains(self.state) # again, if one wishes to train rl agent subject to constraints
        truncated = self.t == self.num_steps

        if (truncated or terminated) and self.episode % 10==0: # somewhat expensive to save episodes---only need periodic snapshots
            self.save_episode()
        
        return {"time": self.t, "distance": distance, "reward": reward, "terminated": terminated, "TimeLimit.truncated": truncated, "full_state": self.state}
    
    def _simulator(self, action):
        # functionally the same as simulator.make_step() in do-mpc but processes the output and dtype
        a = self._fix_action(action)
        x = self.simulator.make_step(a)
        return self._env_state(x)
    
    def _define_simulator(self):
        simulator = self.template_simulator(self.model, uncertain_params=self.uncertain_params, goal=self.goal_map.goal)
        return simulator
    
    def _env_state(self, x0):
        return np.float32(np.reshape(x0, self.observation_space.shape))

    def mpc_state(self, x0):
        return np.float32(np.reshape(x0, self.observation_space.shape[::-1]))
    
    def _fix_action(self, action):
        a = np.reshape(action, self.action_space.shape[::-1])
        return a.reshape(a.shape + (1,))
    
    def renderMPC(self, mpc, num_steps, path=""):
        # quick plotting tool; more refined helpers in the eval
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # Customizing Matplotlib:
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.grid'] = True

        self._run_closed_loop(mpc, self.simulator, num_steps=num_steps)

        fig = do_mpc.graphics.default_plot(self.simulator.data)
        plt.savefig(path)
        plt.close('all')

        return fig[0]

    def _run_closed_loop(self, controller, simulator, num_steps=10, x0=None):

        controller.reset_history()
        simulator.reset_history()
        self.reset()
        x0 = self.state.copy()

        controller.x0 = x0
        simulator.x0 = x0
        controller.set_initial_guess()

        for _ in range(num_steps):
            u0 = controller.make_step(x0)
            x0 = simulator.make_step(u0)
            # do_mpc.tools.printProgressBar(k, num_steps-1, prefix='Closed-loop simulation:', length=50)

        return
    
    def save_episode(self):

        do_mpc.data.save_results([self.simulator], result_name=f"{self.eval}" + "episode-" + f"{self.episode}", result_path=self.path)
        results = do_mpc.data.load_results(self.path+ "/episode-" + f"{self.episode}.pkl")
        if self.episode_plot is not None:
            ep_data = self.episode_plot(results['simulator'], path=self.path)
            plt.savefig(self.path+"/episode-" + f"{self.episode}.png")
            plt.close(ep_data)
        return
        
    def close(self):
        ...

if __name__ == '__main__':
    import os
    import sys
    sys.path.append('')

    # print("Checking CSTR")

    # from envs.CSTR.template_model import template_model
    # from envs.CSTR.template_mpc import template_mpc
    # from envs.CSTR.template_simulator import template_simulator

    # model = template_model()
    # mpc = template_mpc(model, mpc_mode="baseline")

    # # simulator = template_simulator(model)
    # max_x = np.array([2.0,2.0,140.0,140.0]).flatten()
    # min_x = np.array([0.1,0.1,50.0,50.0]).flatten() # writing like this to emphasize do-mpc sizing convention
    # max_u = np.array([10.0,0.0]).flatten()
    # min_u = np.array([0.5,-8.50]).flatten()
    # bounds = {'x_low' : min_x, 'x_high' : max_x, 'u_low' : min_u, 'u_high' : max_u}

    # # run environment
    # gym.register(
    # id="gymnasium_env/DoMPCEnv-v0",
    # entry_point=DoMPCEnv,
    #     )  
    # env = gym.make("gymnasium_env/DoMPCEnv-v0", template_simulator=template_simulator, model=model, bounds=None, same_state=np.array([0.8, 0.4, 134.14, 130.0]))
    # obs,_ = env.reset(seed=0)
    # for _ in range(50):
    #     # a = env.action_space.sample()
    #     a = mpc.make_step(env.mpc_state(obs))
    #     obs, reward, terminated, truncated, info = env.step(a)
    #     print(obs)

    print("Checking LQR")

    from envs.LQR.template_model import template_model
    from envs.LQR.template_mpc import template_mpc
    from envs.LQR.template_simulator import template_simulator

    n, m = 4, 4
    xlim = np.vstack([-1*np.ones(n), 1*np.ones(n)])
    ulim = np.vstack([-1*np.ones(m), 1*np.ones(m)])
    bounds = {'x_low' : xlim[0], 'x_high' : xlim[1], 'u_low' : ulim[0], 'u_high' : ulim[1]}

    model = template_model(n, m)
    mpc = template_mpc(model, mpc_mode="nominal")

    # run environment
    gym.register(
    id="gymnasium_env/DoMPCEnv-v0",
    entry_point=DoMPCEnv,
        )
    same_state = 0.5*np.ones(n) # np.array([0.65, 0.65, 0.652, 0.664])
    env = gym.make("gymnasium_env/DoMPCEnv-v0", template_simulator=template_simulator, model=model, bounds=bounds, same_state=same_state, smooth_reward=False, sa_reward=True)
    obs,_ = env.reset(seed=0)
    
    mpc.x0 = env.mpc_state(obs)
    mpc.set_initial_guess()
    mpc.settings.supress_ipopt_output()
    
    for _ in range(50):
        # a = env.action_space.sample()
        a = mpc.make_step(env.mpc_state(obs))
        obs, reward, terminated, truncated, info = env.step(a)
        print(obs)