import numpy as np
from environment import EnvironmentState
from torch.utils.data import Dataset

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

        arm0.x, arm0.y, v0.x, v0.y, end0.x, end0.y, angular_velocity0,
        arm1.x, arm1.y, v1.x, v1.y, end1.x, end1.y, angular_velocity1,
        on_canvas
        [float, float, float, float, float, float, float, float, float, float, float]
        all input scaled to [-1, 1] with tanh
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

    return lambda d: transform_env_state_to_nn_state(d, include_torque)


class StateDataset(Dataset):
    def __init__(self, env_instance: EnvironmentState, transform_f,
                 size=3500000, skip_step=0, random_torque=True, remove_torque=False):
        self.size = size
        self.skip_step = skip_step
        self.env = env_instance
        self.transform_f = transform_f
        self.random_torque = random_torque
        self.remove_torque = remove_torque
        # initiate environment
        for _ in range(20):
            self.env.step(random_torque=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # set random torque
        if self.random_torque:
            self.env.torque_random_set() # set torque
        if self.remove_torque:
            self.env.step() # get rid of torque
        s0 = self.env.get_current_state()
        for i in range(self.skip_step):
            self.env.step()

        self.env.step(random_torque=False) # random movement when not include torques
        s1 = self.env.get_current_state()

        return {'s0': self.transform_f(s0),
                's1': self.transform_f(s1)}

