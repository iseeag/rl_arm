#!/usr/bin/python

from interface import make_transfrom_f
from environment import EnvironmentState
from actors import ActorP
import torch
import matplotlib


# -------------------------- show action movement in environment plot ------------------------------------------------
if __name__ == '__main__':
    matplotlib.interactive(True)
    env = EnvironmentState()
    config_array = [15, 30, 15, 2]
    actor = ActorP(config_array).load('actor_multi_14_6.pt')
    transform_f = make_transfrom_f()
    env.green_dot([[14.0, 6.0]])  # target point
    env.randomize()
    for i in range(1, 2000):
        env.step()
        if i % 1 == 0:
            with torch.no_grad():
                actions = actor(transform_f(env.get_current_state()))
            env.torque_scaled_set(actions[0], actions[1])
            print(actions, end='\r')
        if i % 100 == 0:
            env.randomize()
        env.plot()
