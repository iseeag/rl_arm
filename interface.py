import numpy as np


def transform_env_to_nn(d):
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
     'torques': [int, int]
     'canvas': np.array(uint8) -> 28 * 28 * 1
     'reachable_distance': int
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

    on_canvas = np.tanh(1) if d['on_canvas'] else np.tanh(-1)
    return np.array([
        arm0.x, arm0.y, v0[0], v0[1], end0[0], end0[1], angular_velocity0,
        arm1.x, arm1.y, v1[0], v1[1], end1[0], end1[1], angular_velocity1,
        on_canvas
    ])