from utils import get_nn_params
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self, input_size, output_size, critic):
        super(Actor, self).__init__()
        self.fc0 = nn.Linear(input_size, input_size*10)
        self.fc1 = nn.Linear(input_size*10, input_size*10)
        self.fc2 = nn.Linear(input_size*10, output_size)
        print(f'number of parameters: {get_nn_params(self)}')

        self.optimizer = optim.Adam(self.parameters())
        self.critic = critic

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def optimize(self, x):
        self.optimizer.zero_grad()
        actions = self(x)
        loss = self.critic(actions).mean()
        loss.backward()
        self.optimizer.step()

        return actions, loss