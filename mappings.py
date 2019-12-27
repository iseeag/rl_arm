import typing
from utils import get_nn_params, add_save_load_optimize_optimizer_optim_context, add_auto_save
from interface import remove_torque, get_torques, add_torque
import torch.nn as nn


# transform what an actor sees
class StateMapStatic(nn.Module):

    def __init__(self, input_features, output_features):
        super(StateMapStatic, self).__init__()
        self.fc0 = nn.Linear(input_features, output_features)
        add_save_load_optimize_optimizer_optim_context(StateMapStatic, self)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        torques = get_torques(x)
        x = remove_torque(x)
        x = self.fc0(x)
        x = add_torque(x, torques)
        return x