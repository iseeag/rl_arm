import typing
from utils import get_nn_params, add_save_load_optimize_optimizer_optim_context, add_auto_save
from interface import remove_torque, get_arm1_end_points
import torch.nn as nn
import torch.nn.functional as F
import torch


class ActorAgregate(nn.Module):

    def __init__(self, state_feature_size, target_feature_size, n_torque_predictors, expand_ratio=1):
        super(ActorAgregate, self).__init__()
        input_size = state_feature_size + target_feature_size
        output_size = n_torque_predictors
        self.fc0 = nn.Linear(input_size, input_size * expand_ratio)
        self.fc1 = nn.Linear(input_size * expand_ratio, output_size)

        n_parameter = get_nn_params(self, True)
        add_save_load_optimize_optimizer_optim_context(ActorAgregate, self)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, s0, s_target, torque_predictors):
        input = torch.cat([remove_torque(s0), get_arm1_end_points(s_target)], -1)
        x = F.relu(self.fc0(input))
        x = self.fc1(x)
        x = torch.softmax(x, -1)

        torques = torch.stack([pr(s0, s_target) for pr in torque_predictors], -2)
        torques = torques * x.unsqueeze(-1)
        torques = torques.sum(-2)

        return torques, x


class Actor(nn.Module):

    def __init__(self, input_size, output_size, critic=None):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc0 = nn.Linear(input_size, input_size*2)
        self.fc1 = nn.Linear(input_size*2, input_size*5)
        self.fc2 = nn.Linear(input_size*5, output_size)
        n_parameter = get_nn_params(self, True)

        add_save_load_optimize_optimizer_optim_context(Actor, self)
        self.critic = critic

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        x = remove_torque(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) # todo: better activation function?
        return x


class ActorP(nn.Module):

    def __init__(self, config_array):
        super(ActorP, self).__init__()
        self.config_array = config_array
        self.in_features = config_array[0]
        self.out_features = config_array[-1]
        fcs = [nn.Linear(i, j) for i, j in zip(config_array[:-1], config_array[1:])]
        self.fc_list = []
        for i, fc in enumerate(fcs):
            setattr(self, f'fc{i}', fc)
            self.fc_list.append(f'fc{i}')

        n_parameter = get_nn_params(self, True)

        add_save_load_optimize_optimizer_optim_context(ActorP, self)
        add_auto_save(ActorP, self)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        x = remove_torque(x)
        for fc_name in self.fc_list[:-1]:
            fc = getattr(self, fc_name)
            x = F.relu(fc(x))
        fc = getattr(self, self.fc_list[-1])
        x = torch.tanh(fc(x)) # todo: better activation function?

        return x


