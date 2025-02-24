import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# neuromancer stuff
from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dynamics import integrators
from neuromancer.constraint import variable
from neuromancer.problem import Problem
from neuromancer.loss import PenaltyLoss
from neuromancer.dataset import DictDataset
from neuromancer.trainer import Trainer
from neuromancer.psl import signals
import torch.optim as optim
from torch.utils.data import DataLoader

class Dynamics(nn.Module):
    def __init__(self, env, rb):
        super().__init__()

        self.env = env
        self.rb = rb

        # Configure network
        self.nx = np.array(env.single_observation_space.shape).prod()
        self.nu = np.array(env.single_action_space.shape).prod()
        self.dx = blocks.ResMLP(self.nx + self.nu, self.nx, bias=True, linear_map=torch.nn.Linear, nonlin=torch.nn.SiLU,
                    hsizes=[64 for h in range(2)])
        self.system_node = Node(self.dx, ['x','u'],['xnext'])
        self.x_shift = Node(lambda x: x, ['xnext'], ['x'])
        self.model = System([self.system_node], nstep_key='u') # or nsteps=1
        self.model_eval = System([self.system_node, self.x_shift], nstep_key='u')

        # Formulate problem
        self.xpred = variable('xnext')
        self.xtrue = variable('xtrue')
        self.loss = (self.xpred == self.xtrue)^2 # confusingly, this refers to a constraint, not a Boolean
        self.obj = PenaltyLoss([self.loss], [])
        self.problem = Problem([self.model], self.obj)

        # Setup optimizer
        self.opt = optim.AdamW(self.model.parameters(), 0.001)

    def forward(self,x,u):
        return self.dx(x,u)
    
    def train(self):
        train_loader = self._train_loader()
        trainer = Trainer(self.problem, train_loader,
                        optimizer=self.opt,
                        epochs=1, epoch_verbose=4,
                        patience=1,
                        train_metric='train_loss', eval_metric='train_loss') # can add a test loss, but the dataset is constantly being updated anyway
        self.best_model = trainer.train() # output is a deepcopy
        return
    
    def _train_loader(self):

        # need to coordinate the number of epoch and batch size
        batch = self.rb.sample(1000)
        data = {}
        data['x'] = batch.observations.unsqueeze(1)
        data['u'] = batch.actions.unsqueeze(1)
        data['xtrue'] = batch.next_observations.unsqueeze(1)
        datadict = DictDataset(data)

        train_loader = DataLoader(datadict, batch_size=64, shuffle=True, collate_fn=datadict.collate_fn)
        return train_loader
    
    def rollout_eval(self):

        ## Add current run logger for wanbd

        obs, info = self.env.reset()
        signal = signals.prbs(10, self.nu, min=self.env.action_space.low, max=self.env.action_space.high, p=.9, rng=np.random.default_rng())/2
        trajectory = self.model_eval({'x': torch.Tensor(obs).unsqueeze(1), 'u':torch.Tensor(signal[None,:])})   
        
        mae = 0.0
        for u,x in zip(signal,trajectory['x'].squeeze().detach().numpy()):
            error = obs.squeeze() - x
            mae += np.abs(np.mean(error))
            obs, rewards, terminations, truncations, infos = envs.step([u])
            # print(s)

        print(mae/len(signal))
        return
        