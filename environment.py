import pymunk as pm
from pymunk import Vec2d
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


class EnvironmentState:
    def __init__(self):
        self.space = pm.Space()
        self.space.damping = 0.9 # to slow down bodies
        # make arms
        self.root0 = (0,0)
        self.arm0 = Arm(self.root0, 24)
        self.root1 = (0,24)
        self.arm1 = Arm(self.root1, 16)
        self.space.add(self.arm0.body,
                       self.arm0.shape,
                       self.arm1.body,
                       self.arm1.shape)
        # set torques
        self.max_torque0 = self.arm0.body.moment
        self.max_torque1 = self.arm1.body.moment * 2 # torque ratio
        self.instant_torques = None
        # join arms
        self.pj0 = pm.PivotJoint(self.arm0.body, self.space.static_body, self.root0)
        self.space.add(self.pj0)
        self.pj1 = pm.PivotJoint(self.arm1.body, self.arm0.body, self.root1)
        self.space.add(self.pj1)
        self.spring_stiffness = 10
        self.drs0 = pm.DampedRotarySpring(self.arm0.body, self.space.static_body, 0, 0, 1)
        self.space.add(self.drs0)
        self.drs1 = pm.DampedRotarySpring(self.arm1.body, self.arm0.body, 0, 0, 1)
        self.space.add(self.drs1)
        # canvas 28 * 28
        self.canvas = Canvas()
        self.cache_point_g = None
        self.cache_point_b = None
        self.cache_point_m = None
        self.cache_point_y = None
        self.cache_point_r = None
        self.cache_point_c = None
        self.cache_point_k = None

    def get_current_state(self):
        return {'arm0': self.arm0.info_dump(),
                'arm1': self.arm1.info_dump(),
                'on_canvas': self.canvas.on_canvas(self.arm1.get_extend_coor()),
                'torques': self.instant_torques,
                'max_torques': [self.max_torque0, self.max_torque1],
                'canvas': self.canvas.canvas,
                'reachable_distance': self.arm0.length + self.arm1.length,
                'energy': (self.arm0.body.kinetic_energy, self.arm1.body.kinetic_energy)
                }

    def draw(self, strength=1):
        x, y = np.round(self.arm1.get_extend_coor()).astype(np.uint8)
        return self.canvas.draw_point((x, y), strength)

    def apply_torque_acw_1(self, torque=5):
        self.drs1.stiffness = self.spring_stiffness
        self.drs1.rest_angle = self.arm1.body.angle - self.arm0.body.angle + torque

    def apply_torque_cw_1(self, torque=5):
        self.apply_torque_acw_1(-torque)

    def apply_torque_acw_0(self, torque=5):
        self.drs0.stiffness = self.spring_stiffness
        self.drs0.rest_angle = self.arm0.body.angle - self.space.static_body.angle + torque

    def apply_torque_cw_0(self, torque=5):
        self.apply_torque_acw_0(-torque)

    def torque_random_set(self):
        m0, m1 = self.max_torque0, self.max_torque1
        self.instant_torques = [np.random.randint(-m0, m0), np.random.randint(-m1, m1)]

    def torque_scaled_set(self, t0, t1):
        m0, m1 = self.max_torque0, self.max_torque1
        self.instant_torques = [m0 * t0, m1 * t1]

    def torque_reset(self):
        self.drs0.stiffness = 0
        self.drs1.stiffness = 0

    def randomize(self):
        self.arm0.body.angle = (1 - 2 * np.random.rand()) * 6.28
        self.arm1.body.angle = (1 - 2 * np.random.rand()) * 6.28
        for _ in range(500): self.step()

    def step(self, actions=None, ds=1/8, random_torque=False):
        assert(actions is None or not random_torque) # either or none but not both
        # apply previous torques
        if self.instant_torques is not None:
            self.apply_torque_cw_0(self.instant_torques[0])
            self.apply_torque_cw_1(self.instant_torques[1])
            self.instant_torques = None

        if random_torque:
            # set torques for next step
            if np.random.randint(8) == 0:
                self.torque_random_set()
            self.space.step(ds)
            self.torque_reset()

        elif actions is None:
            self.space.step(ds)
            self.torque_reset()

        else:
            for a in actions:
                a()
            self.space.step(ds)
            self.torque_reset()

    def plot(self):
        plt.cla()
        self.newline(self.arm0.get_root_coor(), self.arm0.get_extend_coor())
        self.newline(self.arm1.get_root_coor(), self.arm1.get_extend_coor())
        self.new_dots(self.arm0.get_root_coor(),
                      self.arm0.get_extend_coor(),
                      self.arm1.get_extend_coor())
        self.draw_color_dot()

        plt.show()
        plt.pause(0.001)

    def loop_plot(self, t=10, random_torque=False):
        print(f'steps and loop for {t}s')
        t0 = time.time()
        while t0 + t > time.time():
            self.step(random_torque=random_torque)
            self.plot()
            # plt.pause(0.001)

    def newline(self, p1, p2):
        plt.ylim(-50, 50)
        plt.xlim(-50, 50)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b')


    def new_dots(self, *point, color='r'):
        plt.plot([p[0] for p in point], [p[1] for p in point], f'{color}o')

    def draw_color_dot(self):
        for p, c in [(self.cache_point_g, 'g'),
                     (self.cache_point_y, 'y'),
                     (self.cache_point_m, 'm'),
                     (self.cache_point_b, 'b'),
                     (self.cache_point_r, 'r'),
                     (self.cache_point_c, 'c'),
                     (self.cache_point_k, 'k')]:
            if p is not None: self.new_dots(*p, color=c)

    def green_dot(self, point):
        self.cache_point_g = point

    def red_dot(self, point):
        self.cache_point_r = point

    def blue_dot(self, point):
        self.cache_point_b = point

    def yellow_dot(self, point):
        self.cache_point_y = point

    def magenta_dot(self, point):
        self.cache_point_m = point

    def cyan_dot(self, point):
        self.cache_point_c = point

    def black_dot(self, point):
        self.cache_point_k = point

    def reset(self):
        self = EnvironmentState()

class Arm:
    def __init__(self, pivot_coor, length, angle=0):
        self.length = length
        self.low_end_coor = pivot_coor
        self.high_end_coor = (pivot_coor[0], pivot_coor[1]+length)
        self.body = pm.Body()
        self.body.position = Vec2d(pivot_coor[0], pivot_coor[1]+length/2)
        self.shape = pm.Segment(self.body, self.low_end_coor, self.high_end_coor, 0.5)
        self.shape.mass = self.length

    def get_root_coor(self):
        return self.body.local_to_world((0, -self.length/2))

    def get_extend_coor(self):
        return self.body.local_to_world((0, self.length/2))

    def info_dump(self):
        return {'root': self.body.local_to_world((0, -self.length/2)),
                'end': self.body.local_to_world((0, self.length/2)),
                'velocity': self.body.velocity,
                'angle': self.body.angle,
                'angular_velocity': self.body.angular_velocity}

    def apply_pinch_cw(self, torque=5):
        self.body.apply_force_at_local_point((torque, 0), (0, -self.length/2+1))
        self.body.apply_force_at_local_point((-torque, 0), (0,-self.length/2))

    def apply_pinch_acw(self, torque=5):
        self.apply_pinch_cw(-torque)

class Canvas:
    def __init__(self, width=28, height=28):
        self.width = width
        self.height = height
        self.canvas = np.zeros((self.width, self.height, 1), np.uint8)

    def on_canvas(self, point):
        x, y = point[0], point[1]
        return 0 <= x and x <= self.width - 1 and 0 <= y and y <= self.height - 1

    def draw_point(self, point, strength):
        if self.on_canvas(point):
            s = self.canvas[point] + strength
            self.canvas[point] = s if s >= self.canvas[point] else 255
            return [1, 0] # signal on canvas
        else:
            return [0, 1] # signal off canvas

    def reset(self):
        self.canvas = np.zeros((self.width, self.height, 1), np.uint8)

    def display(self, t=5):
        # cv2.imshow('bl', cv2.flip(self.canvas, 1))
        cv2.imshow('bl', cv2.flip(cv2.transpose(self.canvas), 0))
        cv2.waitKey(t*1000)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    env_state = EnvironmentState()
    # env_state.step([
    #     # lambda: env_state.arm0.apply_torque_acw(200),
    #     # lambda: env_state.arm1.apply_pinch_cw(200),
    #     lambda: env_state.apply_torque_cw_1(2000),
    #     # lambda: env_state.apply_torque_cw_0(2000),
    # ])
    env_state.loop_plot(t=1000, random_torque=True)
    for i in range(10):
        env_state.step(random_torque=True)
        print(env_state.get_current_state()[1], env_state.get_current_state()[2])

    s = env_state.get_current_state()