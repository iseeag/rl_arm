import numpy as np
from environment import EnvironmentState
from collections import deque
from torch.utils.data import Dataset
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

        arm0.x, arm0.y, v0.x, v0.y, end0.x, end0.y, angular_velocity0, torque0[optional],
        arm1.x, arm1.y, v1.x, v1.y, end1.x, end1.y, angular_velocity1, torque1[optional],
        on_canvas
        [float] * 17 (with torque)
        [float] * 15 (without torque)
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


def replace_torque(x: Tensor, a: Tensor): # [float]*17, [float]*2
    return torch.cat((x.narrow(-1, 0, 7), a.narrow(-1, 0, 1),
                      x.narrow(-1, 8, 7), a.narrow(-1, 1, 1),
                      x.narrow(-1, 16, 1)), dim=-1)


def get_arm1_end_points(x: Tensor):
    return x.narrow(-1, 12, 2)


def get_torques(x: Tensor):
    return torch.cat([x.narrow(-1, 7, 1),x.narrow(-1, 15, 1)], dim=-1)

def fill_arm1_endpoint(endpoints: Tensor):
    shape = list(endpoints.shape)
    shape[-1] = 17
    zeros = torch.zeros(shape)
    return torch.cat((zeros.narrow(-1, 0, 12), endpoints, zeros.narrow(-1, 14, 3)), dim=-1)

class StateDataset(Dataset):

    def __init__(self, env_instance: EnvironmentState, size=3500000, skip_step=0,
                 random_torque=True, remove_torque=False):
        self.size = size
        self.skip_step = skip_step
        self.env = env_instance
        self.transform_f = make_transfrom_f(include_torque=True)
        self.output_size = len(self.transform_f(self.env.get_current_state()))
        self.output_size_for_actor = self.output_size - 2
        self.random_torque = random_torque
        self.remove_torque = remove_torque
        # initiate environment
        for _ in range(20):
            self.env.step(random_torque=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # set random torque
        if self.random_torque and idx % 2 == 0:
            for _ in range(self.skip_step//8 + 1):
                self.env.torque_random_set()
                self.env.step()
        if self.remove_torque:
            self.env.step() # get rid of torque
        s0 = self.env.get_current_state()
        for i in range(self.skip_step):
            self.env.step()

        self.env.step(random_torque=False)
        s1 = self.env.get_current_state()

        return {'s0': self.transform_f(s0),
                's1': self.transform_f(s1)}

    def set_torque(self, t0, t1):
        self.env.torque_scaled_set(t0, t1)



class StateDatasetLSTMTorque(Dataset):
    '''Dataset for PredictorLSTMTorque'''
    def __init__(self, env_instance: EnvironmentState, size=3500000, skip_step=1, seq_len=15):
        self.size = size
        self.skip_step = skip_step
        self.seq_leng = seq_len
        self.env = env_instance
        self.transform_f = make_transfrom_f(include_torque=True)
        self.output_size = len(self.transform_f(self.env.get_current_state())) - 2
        self.remove_torque = remove_torque
        # initiate environment
        self.cache_result = deque(maxlen=self.seq_leng + 1)
        for _ in range(20):
            self.env.step(random_torque=True)
        self.move_and_cache()


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        for i in range(self.skip_step):
            self.env.step()
        self.env.torque_random_set()
        self.append_result()

        state_list = torch.stack(list(self.cache_result))
        s0 = state_list[0]
        torques = get_torques(state_list[:-1])
        diffs = get_arm1_end_points(state_list)[1:] - get_arm1_end_points(state_list)[:-1]


        return {'s0': s0,
                'torques': torques,
                'diffs': diffs}

    def append_result(self):
        s = self.transform_f(self.env.get_current_state())
        self.cache_result.append(s)

    def move_and_cache(self):
        for _ in range(4): # get some energy
            self.env.step()
            self.env.torque_random_set()
        for i in range(self.skip_step * self.seq_leng):
            if i % self.skip_step == 0:
                self.append_result()
            self.env.step()



class StateDatasetLSTM(Dataset):

    def __init__(self, env_instance: EnvironmentState, size=3500000, skip_step=4, seq_leng=10, remove_torque=False):
        self.size = size
        self.skip_step = skip_step
        self.seq_leng = seq_leng
        self.env = env_instance
        self.transform_f = make_transfrom_f(include_torque=True)
        self.output_size = len(self.transform_f(self.env.get_current_state()))
        self.output_size_for_actor = self.output_size - 2
        self.remove_torque = remove_torque
        # initiate environment
        self.cache_result = deque(maxlen=self.seq_leng)
        for _ in range(20):
            self.env.step(random_torque=True)
        self.move_and_cache()


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx % 9 == 0: # new action every ninth step
            self.move_and_cache()

        if self.remove_torque:
            for i in range(self.skip_step):
                self.env.step() # get rid of torque
            self.append_result() # append env state to cache deque

        s0 = self.cache_result.popleft()
        for i in range(self.skip_step):
            self.env.step()
        self.append_result()

        return {'s0': s0,
                'result': torch.stack(list(self.cache_result))}

    def append_result(self):
        s1 = self.transform_f(self.env.get_current_state())
        self.cache_result.append(s1)

    def move_and_cache(self):
        for _ in range(4): # torque * 4 to ramp up movement
            self.env.step()
            self.env.torque_random_set()
        for i in range(self.skip_step * self.seq_leng):
            if i % self.skip_step == 0:
                self.append_result()
            self.env.step()

    def set_torque(self, t0, t1):
        self.env.torque_scaled_set(t0, t1)


class RewardDataset(Dataset):
    def __init__(self, env_instance: EnvironmentState, size):
        self.env = env_instance
        self.size = size
        ...

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # set random torque
        if self.random_torque and idx % 2 == 0:
            for _ in range(self.skip_step // 8 + 1):
                self.env.torque_random_set()
                self.env.step()
        if self.remove_torque:
            self.env.step()  # get rid of torque
        s0 = self.env.get_current_state()
        for i in range(self.skip_step):
            self.env.step()

        self.env.step(random_torque=False)
        s1 = self.env.get_current_state()

        return {'s0': self.transform_f(s0),
                's1': self.transform_f(s1)}