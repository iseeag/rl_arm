import numpy as np
from torch.utils.data import Dataset

def transform_state_env_to_nn(d):
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
    # arm0
    arm0 = (d['arm0']['end'] - d['arm0']['root']).normalized()
    v0 = np.tanh(d['arm0']['velocity'])
    end0 = d['arm0']['end'] / d['reachable_distance']
    angular_velocity0 = np.tanh(d['arm0']['angular_velocity'])
    # arm1
    arm1 = (d['arm1']['end'] - d['arm1']['root']).normalized()
    v1 = np.tanh(d['arm1']['velocity'])
    end1 = d['arm1']['end'] / d['reachable_distance']
    angular_velocity1 = np.tanh(d['arm1']['angular_velocity'])
    # on_canvas
    on_canvas = np.tanh(1) if d['on_canvas'] else np.tanh(-1)

    return np.array([
        arm0.x, arm0.y, v0[0], v0[1], end0[0], end0[1], angular_velocity0,
        arm1.x, arm1.y, v1[0], v1[1], end1[0], end1[1], angular_velocity1,
        on_canvas
    ], dtype=np.float32)


def transform_env_state_to_nn_with_torques(d):
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
    torque: normalized by max torque

    arm0.x, arm0.y, v0.x, v0.y, end0.x, end0.y, angular_velocity0, torque0,
    arm1.x, arm1.y, v1.x, v1.y, end1.x, end1.y, angular_velocity1, torque1
    on_canvas
    [float, float, float, float, float, float, float, float, float, float, float]
    all input scaled to [-1, 1] with tanh
    """
    # arm0
    arm0 = (d['arm0']['end'] - d['arm0']['root']).normalized()
    v0 = np.tanh(d['arm0']['velocity'])
    end0 = d['arm0']['end'] / d['reachable_distance']
    angular_velocity0 = np.tanh(d['arm0']['angular_velocity'])
    torque0 = d['torques'][0] / d['max_torques'][0] # todo: need to be discrete

    # arm1
    arm1 = (d['arm1']['end'] - d['arm1']['root']).normalized()
    v1 = np.tanh(d['arm1']['velocity'])
    end1 = d['arm1']['end'] / d['reachable_distance']
    angular_velocity1 = np.tanh(d['arm1']['angular_velocity'])
    torque0 = d['torques'][1] / d['max_torques'][1] # todo: need to be discrete

    # on_canvas
    on_canvas = np.tanh(1) if d['on_canvas'] else np.tanh(-1)

    return np.array([
        arm0.x, arm0.y, v0[0], v0[1], end0[0], end0[1], angular_velocity0,
        arm1.x, arm1.y, v1[0], v1[1], end1[0], end1[1], angular_velocity1,
        on_canvas
    ], dtype=np.float32)


class StateDataset(Dataset):
    def __init__(self, env_instance, size=3500000, skip_step=0):
        self.size = size
        self.skip_step = skip_step
        self.env = env_instance
        # initiate environment
        for _ in range(20):
            self.env.step(random_movement=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        while True:
            s0 = self.env.get_current_state()
            torques_record = s0['torques']
            if torques_record is not None:
                self.env.step(random_movement=True)
                continue # found torques start all over again
            # skip a steps
            for i in range(self.skip_step):
                self.env.step(random_movement=True)
                torques_record = self.env.get_current_state()['torques']
                if torques_record is not None: break # found torques breakout for loop
            if torques_record is not None: continue # found torques start over again

            # no torques found, breakout while loop
            break

        self.env.step(random_movement=True)
        s1 = self.env.get_current_state()

        return {'s0': transform_state_env_to_nn(s0),
                's1': transform_state_env_to_nn(s1)}

