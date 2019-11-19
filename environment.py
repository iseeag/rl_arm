import pymunk as pm
from pymunk import Vec2d
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time
import numpy as np
import cv2


class EnvironmentState:
    def __init__(self):
        self.space = pm.Space()
        self.space.damping = 0.9 # to slow down bodies
        # make arms
        self.root0 = (0,0)
        self.arm0 = Arm(self.root0, 24)
        self.root1 = (0,24)
        self.arm1 = Arm(self.root1, 18)
        self.space.add(self.arm0.body,
                       self.arm0.shape,
                       self.arm1.body,
                       self.arm1.shape)
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

    def get_current_state(self):
        return self.arm0.info_dump(), self.arm1.info_dump(), self.canvas.canvas

    def draw(self, strength=1):
        x, y = np.round(self.arm1.get_extend_coor()).astype(np.uint8)
        return self.canvas.draw_point((x, y), strength)

    def apply_torque_acw_1(self, torque=5):
        self.drs1.stiffness = self.spring_stiffness
        self.drs1.rest_angle = self.arm1.body.angle - self.arm0.body.angle + torque/3.14

    def apply_torque_cw_1(self, torque=5):
        self.apply_torque_acw_1(-torque)

    def apply_torque_acw_0(self, torque=5):
        self.drs0.stiffness = self.spring_stiffness
        self.drs0.rest_angle = self.arm0.body.angle - self.space.static_body.angle + torque / 3.14

    def apply_torque_cw_0(self, torque=5):
        self.apply_torque_acw_0(-torque)

    def torque_reset(self):
        self.drs0.stiffness = 0
        self.drs1.stiffness = 0

    def step(self, actions=None, ds=1/8):
        if actions is None:
            self.space.step(ds)
        else:
            for a in actions:
                a()
            self.space.step(ds)
            self.torque_reset()

    def plot(self):
        plt.cla()
        self.newline(self.arm0.get_root_coor(), self.arm0.get_extend_coor())
        self.newline(self.arm1.get_root_coor(), self.arm1.get_extend_coor())
        plt.show()

    def loop_plot(self, t=10):
        print(f'steps and loop for {t}s')
        t0 = time.time()
        while t0 + t > time.time():
            self.step()
            self.plot()
            plt.pause(0.001)

    def newline(self, p1, p2):
        plt.ylim(-50, 50)
        plt.xlim(-50, 50)
        ax = plt.gca()

        xmin, xmax = ax.get_xbound()

        if (p2[0] == p1[0]):
            xmin = xmax = p1[0]
            ymin, ymax = ax.get_ybound()
        else:
            ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
            ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

        l = mlines.Line2D([xmin, xmax], [ymin, ymax])
        ax.add_line(l)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'ro')

    def reset(self):
        ...

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
        return (self.body.local_to_world((0, -self.length/2)),
                self.body.local_to_world((0, self.length/2)),
                self.body.velocity,
                self.body.angle,
                self.body.angular_velocity)

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

env_state = EnvironmentState()
env_state.step([
    # lambda: env_state.arm0.apply_torque_acw(200),
    # lambda: env_state.arm1.apply_pinch_cw(200),
    lambda: env_state.apply_torque_cw_1(2000),
    # lambda: env_state.apply_torque_cw_0(2000),
])
env_state.loop_plot(2)