#!/usr/bin/python

from interface import make_transfrom_f
from dataset import StateDataset
from environment import EnvironmentState
from actors import Actor, ActorP, ActorAgregate
import torch
import matplotlib

# -------------------------- show action movement in environment plot --------------------------------------------
if __name__ == '__main__':
    matplotlib.interactive(True)
    env = EnvironmentState()
    state_set = StateDataset(env, skip_step=0, size=4000000, random_torque=True, remove_torque=True)
    actor = Actor(state_set.output_size_for_actor, 2).load('actor_1.pt')
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
