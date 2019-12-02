from utils import get_nn_params, OptimizerContext
from interface import remove_torque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

class Actor(nn.Module):

    def __init__(self, input_size, output_size, critic=None):
        super(Actor, self).__init__()
        self.save_path = 'saved_models'
        self.input_size = input_size # state without torque
        self.output_size = output_size # 2 floats represents torques of both arms
        self.fc0 = nn.Linear(input_size, input_size*2)
        self.fc1 = nn.Linear(input_size*2, input_size*5)
        self.fc2 = nn.Linear(input_size*5, output_size)
        print(f'number of parameters: {get_nn_params(self)}')

        self.optimizer = optim.Adam(self.parameters())
        self.optim_cont = OptimizerContext(self.optimizer)
        self.critic = critic

    def forward(self, x):
        x = remove_torque(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) # todo: better activation function?
        return x


    def optimize(self):
        return self.optim_cont

    def save(self, name):
        torch.save(self.state_dict(), f'{self.save_path}/{name}')
        print(f'{self.save_path}/{name} saved')

    def load(self, name):
        self.load_state_dict(torch.load(f'{self.save_path}/{name}'))
        self.eval()
        print('load successful')
        return True


