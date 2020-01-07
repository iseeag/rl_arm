import numpy as np
from torch import Tensor
import torch


def make_transfrom_f(include_torque=True):
    """
        environment state schema:
        {'arm0':
            {'root': Vec2D[float, float]
             'end': Vec2D[float, float]
             'velocity': Vec2D[float, float]
             'angle': float,
             'angular_velocity': float}
            'arm1':
                {'root': Vec2D[float, float]
                'end': Vec2D[float, float]
                'velocity': Vec2D[float, float]
                'angle': float,
                'angular_velocity': float}
         'on_canvas': bool
         'torques': [int, int] | None
         'max_torques': [float, float]
         'canvas': np.array(uint8) -> 28 * 28 * 1
         'reachable_distance': int
         'energy': (float, float)
         }

        Neural network input schema:

        arm0: normalized direction vector
        v0: center velocity vector
        end: end position normalized by reachable_distance

        [float] * 17 (with torque)
        [0]arm0.x, [1]arm0.y,  [2]v0.x,  [3]v0.y,  [4]end0.x,  [5]end0.y,  [6]angular_velocity0,  [7]torque0[optional],
        [8]arm1.x, [9]arm1.y, [10]v1.x, [11]v1.y, [12]end1.x, [13]end1.y, [14]angular_velocity1, [15]torque1[optional],
        [16]on_canvas

        [float] * 15 (without torque)
        [0]arm0.x, [1]arm0.y, [2]v0.x,  [3]v0.y,  [4]end0.x,  [5]end0.y,  [6]angular_velocity0,
        [7]arm1.x, [8]arm1.y, [9]v1.x, [10]v1.y, [11]end1.x, [12]end1.y, [13]angular_velocity1
        [14]on_canvas

        all input scaled to [-1, 1] with tanh except end[0,1].[x,y]
        """
    def transform_env_state_to_nn_state(d, include_torque):
        # arm0
        arm0 = (d['arm0']['end'] - d['arm0']['root']).normalized()
        v0 = np.tanh(d['arm0']['velocity'])
        end0 = d['arm0']['end']
        angular_velocity0 = np.tanh(d['arm0']['angular_velocity'])
        # todo: propose discrete version
        torque0 = d['torques'][0] / d['max_torques'][0] if d['torques'] is not None else 0.0
        # arm1
        arm1 = (d['arm1']['end'] - d['arm1']['root']).normalized()
        v1 = np.tanh(d['arm1']['velocity'])
        end1 = d['arm1']['end']
        angular_velocity1 = np.tanh(d['arm1']['angular_velocity'])
        # todo: propose discrete version
        torque1 = d['torques'][1] / d['max_torques'][1] if d['torques'] is not None else 0.0
        # on_canvas
        on_canvas = np.tanh(1) if d['on_canvas'] else np.tanh(-1)

        if include_torque:
            return np.array([
                arm0.x, arm0.y, v0[0], v0[1], end0[0], end0[1], angular_velocity0, torque0,
                arm1.x, arm1.y, v1[0], v1[1], end1[0], end1[1], angular_velocity1, torque1,
                on_canvas
            ], dtype=np.float32)
        else:
            return np.array([
                arm0.x, arm0.y, v0[0], v0[1], end0[0], end0[1], angular_velocity0,
                arm1.x, arm1.y, v1[0], v1[1], end1[0], end1[1], angular_velocity1,
                on_canvas
            ], dtype=np.float32)

    return lambda d: torch.tensor(transform_env_state_to_nn_state(d, include_torque))


def remove_torque(x: Tensor):
    return torch.cat((x.narrow(-1, 0, 7), x.narrow(-1, 8, 7), x.narrow(-1, 16, 1)), dim=-1)


def add_torque(x: Tensor, a: Tensor):
    return torch.cat((x.narrow(-1, 0, 7), a.narrow(-1, 0, 1),
                      x.narrow(-1, 7, 7), a.narrow(-1, 1, 1),
                      x.narrow(-1, 14, 1)), dim=-1)


def replace_torque(x: Tensor, a: Tensor): # [float]*17, [float]*2
    return torch.cat((x.narrow(-1, 0, 7), a.narrow(-1, 0, 1),
                      x.narrow(-1, 8, 7), a.narrow(-1, 1, 1),
                      x.narrow(-1, 16, 1)), dim=-1)


def replace_arm1_endpoint(x: Tensor, e: Tensor): # [float]*17, [float]*2
    return torch.cat((x.narrow(-1, 0, 12), e.narrow(-1, 0, 2),
                      x.narrow(-1, 14, 3)), dim=-1)


def get_arm1_end_points(x: Tensor):
    return x.narrow(-1, 12, 2)


def get_torques(x: Tensor):
    return torch.cat([x.narrow(-1, 7, 1),x.narrow(-1, 15, 1)], dim=-1)


def fill_arm1_endpoint(endpoints: Tensor):
    shape = list(endpoints.shape)
    shape[-1] = 17
    zeros = torch.zeros(shape)
    return torch.cat((zeros.narrow(-1, 0, 12), endpoints, zeros.narrow(-1, 14, 3)), dim=-1)

