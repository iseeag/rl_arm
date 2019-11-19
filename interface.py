f'''

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
 }

Neural network input schema:

arm0: normalized direction vector
v0: center velocity vector

arm0.x, arm0.y, v0.x, v0.y, angle0, angular_velocity0, 
arm1.x, arm1.y, v1.x, v1.y, angle1, angular_velocity1,
on_canvas
[float, float, float, float, float, float, float, float, float, float, float, float, float]
all input scaled to [-1, 1] with tanh

'''
