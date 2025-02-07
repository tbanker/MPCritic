import torch
import torch.nn as nn

kwargs = {'dtype' : torch.float32,
          'device' : 'cpu'}

class TestModel(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = nn.Parameter(torch.from_numpy(p))

    def forward(self, x):
        return self.p*x
    
class QuadraticStageCost(nn.Module):
    def __init__(self, n, m, Q=None, R=None):
        super().__init__()
        Q = torch.rand((n,n), **kwargs) if Q is None else torch.from_numpy(Q)
        R = torch.rand((m,m), **kwargs) if R is None else torch.from_numpy(R)
        self.n = n
        self.m = m
        self.Q = nn.Parameter(Q)
        self.R = nn.Parameter(R)
        # Q, R share memory with original numpy arrays

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
        # P shares memory with original numpy array

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
        # A, B share memory with original numpy arrays

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
        # K shares memory with original numpy array

    def forward(self, input):
        """ Kx """
        x = input
        return x @ self.K.T
    
if __name__ == '__main__':
    import numpy as np

    b, n, m = 3, 4, 2
    x, u = torch.ones((b,n), **kwargs), torch.ones((b,m), **kwargs)
    x[:,0], u[:,0] = 2., 2.
    z = torch.concat((x,u), dim=-1)
    
    Q, R = np.ones((n,n)), np.ones((m,m))
    stage_cost = QuadraticStageCost(n, m, Q, R)
    term_cost = QuadraticTerminalCost(n, Q)
    
    A, B = np.ones((n,n)), np.ones((n,m))
    dynamics = LinearDynamics(n, m, A, B)

    K = np.ones((m,n))
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

