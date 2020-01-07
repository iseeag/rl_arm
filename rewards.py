from utils import get_nn_params, add_save_load_optimize_optimizer_optim_context
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F



'''
        Neural network input schema:

        arm0: normalized direction vector
        v0: center velocity vector
        end: end position normalized by reachable_distance

        arm0.x, arm0.y, v0.x, v0.y, end0.x, end0.y, angular_velocity0, torque0[optional],
        arm1.x, arm1.y, v1.x, v1.y, end1.x, end1.y, angular_velocity1, torque1[optional],
        on_canvas
        [float, float, float, float, float, float, float, float, float, float, float]
        all input scaled to [-1, 1] with tanh except end[0,1].[x,y]
'''

# a specific implementation
def get_endpoint1(s: Tensor):
    return s.index_select(-1, torch.tensor([12,13]))

def get_endpoint1x(s: Tensor):
    return s.index_select(-1, torch.tensor([12]))

def get_endpoint1y(s: Tensor):
    return s.index_select(-1, torch.tensor([13]))

# incremental rewards
def reward_f(s0: Tensor, s1: Tensor):
    target = torch.tensor([0.0, -20.0])
    d0 = target.dist(get_endpoint1(s0))
    d1 = target.dist(get_endpoint1(s1))
    return (d0 - d1).mean()

def reward_f1(s0: Tensor, s1: Tensor):
    target = torch.tensor([14.0, 6])
    d0x = torch.abs(get_endpoint1x(s0) - target[0])
    d0y = torch.abs(get_endpoint1y(s0) - target[1])
    d1x = torch.abs(get_endpoint1x(s1) - target[0])
    d1y = torch.abs(get_endpoint1y(s1) - target[1])
    return (d0x - d1x + d0y - d1y).mean()

def reward_f2(s0: Tensor, s1: Tensor, target=torch.tensor([14.0, 6])): # multi point
    # s0, s1: (batch_size, seq_len, input_size)
    d0 = torch.abs(get_endpoint1(s0) - target)
    d1 = torch.abs(get_endpoint1(s1) - target)
    return (d0 - d1).mean()


# distance shortening rewards
def reward_dist_reduce_f1(s0: Tensor, target=torch.tensor([14.0, 6])):
    return - torch.abs(get_endpoint1(s0) - target)


def reward_dist_reduce_f2(s0: Tensor, target=torch.tensor([14.0, 6])):
    return - target.dist(get_endpoint1(s0))


def dist_f(s0: Tensor, s1: Tensor):
    return get_endpoint1(s0).dist(get_endpoint1(s1))


class Rewarder(nn.Module):

    def __init__(self, in_size, out_size):
        super(Rewarder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc0 = nn.Linear(in_size, in_size*3)
        self.fc1 = nn.Linear(in_size*3, in_size*6)
        self.fc2 = nn.Linear(in_size*6, out_size)
        print(f'number of parameters: {get_nn_params(self)}')

        add_save_load_optimize_optimizer_optim_context(Rewarder, self)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
