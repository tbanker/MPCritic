import torch
import torch.nn as nn
import numpy as np

kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}        

class GoalMap(nn.Module):
    def __init__(self, idx:list=None, goal=0.0, learn_goal=False):
        super().__init__()
        self.idx = idx
        if idx is not None:
            self.ny = len(idx)
        else:
            self.ny = None

        if learn_goal:
            self.goal = nn.Parameter(torch.Tensor([goal]), requires_grad=True)
        else:
            self.goal = goal

    def forward(self, input):
        if self.idx is None:
            return input
        else:
            return input[...,self.idx] - self.goal
    
    def forward_env(self, x):
        return x[self.idx] - self.goal

class GoalConditionedStageCost(nn.Module):
    def __init__(self, n, m, std=0.5):
        super().__init__()
        self.n = n
        self.m = m
        self.std = std

    def forward(self, input):
        """ -e^(-0.5*input^2) """
        # L4CasADi models can only have 1 input, not (x,u)
        x, u = input[..., :self.n], input[..., self.n:]
        return -torch.exp(-0.5*(x**2).sum(axis=1, keepdims=True) / self.std**2)

class QuadraticStageCost(nn.Module):
    def __init__(self, n, m, Q=None, R=None):
        super().__init__()
        Q = torch.rand((n,n), **kwargs) if Q is None else torch.from_numpy(Q)
        R = torch.rand((m,m), **kwargs) if R is None else torch.from_numpy(R)
        self.n = n
        self.m = m
        self.Q = nn.Parameter(Q)
        self.R = nn.Parameter(R)

    def forward(self, input):
        """ x^TQx + u^TRu """
        # L4CasADi models can only have 1 input, not (x,u)
        x, u = input[..., :self.n], input[..., self.n:]
        return (x @ self.Q * x).sum(axis=1, keepdims=True) + (u @ self.R * u).sum(axis=1, keepdims=True)  

class QuadraticTerminalCost(nn.Module):
    def __init__(self, n, P=None):
        super().__init__()
        P = torch.rand((n,n), **kwargs) if P is None else torch.from_numpy(P)
        self.n = n
        self.P = nn.Parameter(P)

    def forward(self, input):
        """ x^TPx """
        x = input
        return (x @ self.P * x).sum(axis=1, keepdims=True)
    
class LinearDynamics(nn.Module):
    def __init__(self, n, m, A=None, B=None):
        super().__init__()
        A = torch.rand((n,n), **kwargs) if A is None else torch.from_numpy(A)
        B = torch.rand((n,m), **kwargs) if B is None else torch.from_numpy(B)
        self.n = n
        self.m = m
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

    def forward(self, input):
        """ Ax + Bu """
        # L4CasADi models can only have 1 input, not (x,u)
        x, u = input[..., :self.n], input[..., self.n:]
        return x @ self.A.T + u @ self.B.T
    
class LinearPolicy(nn.Module):
    def __init__(self, n, m, K=None):
        super().__init__()
        K = torch.rand((m,n), **kwargs) if K is None else torch.from_numpy(K)
        self.n = n
        self.m = m
        self.K = nn.Parameter(K)

    def forward(self, input):
        """ Kx """
        x = input
        return x @ self.K.T

class PDQuadraticStageCost(nn.Module):
    def __init__(self, n, m, N=None, M=None, epsilon=0.001):
        super().__init__()
        N = torch.rand((n,n), **kwargs) 
        M = torch.rand((m,m), **kwargs)
        self.n = n
        self.m = m
        self.N = nn.Parameter(N)
        self.M = nn.Parameter(M)
        self.epsilon = epsilon

    @property
    def Q(self):
        return self.N.T @ self.N + self.epsilon*torch.eye(self.n)
    
    @property
    def R(self):
        return self.M.T @ self.M + self.epsilon*torch.eye(self.m)

    def forward(self, input):
        """ x^TQx + u^TRu """
        # L4CasADi models can only have 1 input, not (x,u)
        x, u = input[..., :self.n], input[..., self.n:]
        return (x @ self.Q * x).sum(axis=1, keepdims=True) + (u @ self.R * u).sum(axis=1, keepdims=True)

class PDQuadraticTerminalCost(nn.Module):
    def __init__(self, n, L=None, epsilon=0.001):
        super().__init__()
        L = torch.rand((n,n), **kwargs) if L is None else torch.from_numpy(L)
        self.n = n
        self.L = nn.Parameter(L,requires_grad=True)
        self.epsilon = epsilon

    @property
    def P(self):
        return self.L.T @ self.L + self.epsilon*torch.eye(self.n)

    def forward(self, input):
        """ x^TPx """
        x = input
        return (x @ self.P * x).sum(axis=1, keepdims=True)

if __name__ == '__main__':
    import numpy as np
    np_kwargs = {'dtype' : np.float32}

    b, n, m = 3, 4, 2
    x, u = torch.ones((b,n), **kwargs), torch.ones((b,m), **kwargs)
    x[:,0], u[:,0] = 2., 2.
    z = torch.concat((x,u), dim=-1)
    
    Q, R = np.ones((n,n), dtype=np_kwargs['dtype']), np.ones((m,m), dtype=np_kwargs['dtype'])
    stage_cost = QuadraticStageCost(n, m, Q, R)
    term_cost = QuadraticTerminalCost(n, Q)
    
    A, B = np.ones((n,n), dtype=np_kwargs['dtype']), np.ones((n,m), dtype=np_kwargs['dtype'])
    dynamics = LinearDynamics(n, m, A, B)

    K = np.ones((m,n), dtype=np_kwargs['dtype'])
    policy = LinearPolicy(n, m, K)

    res = {
        'x' : x,
        'u' : u,
        'Stage Cost' : stage_cost(z),
        'Terminal Cost' : term_cost(x),
        'Dynamics' : dynamics(z),
        'Policy' : policy(x),
    }

    for key, value in res.items():
        print(f'{key}: {value}')