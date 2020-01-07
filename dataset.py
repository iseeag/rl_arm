from interface import *
from environment import EnvironmentState
from collections import deque
from torch.utils.data import Dataset
import torch


class StateDatasetOfTorque(Dataset):

    def __init__(self, env_instance: EnvironmentState, size=3500000, skip_step=0,
                 random_torque=True, remove_torque=False):
        self.size = size
        self.skip_step = skip_step
        self.env = env_instance
        self.transform_f = make_transfrom_f(include_torque=True)
        self.output_size = self.transform_f(self.env.get_current_state()).shape[0]
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
            for _ in range(5):
                self.env.step(random_torque=True)
        self.env.torque_random_set()
        if self.remove_torque:
            self.env.step() # get rid of torque
        s0 = self.env.get_current_state()
        for i in range(self.skip_step):
            self.env.step()

        self.env.step(random_torque=False)
        s1 = self.env.get_current_state()

        return {'s0': self.transform_f(s0),
                's1': self.transform_f(s1),
                't0': get_torques(self.transform_f(s0))}


class StateDataset(Dataset):

    def __init__(self, env_instance: EnvironmentState, size=3500000, skip_step=0,
                 random_torque=True, remove_torque=False):
        self.size = size
        self.skip_step = skip_step
        self.env = env_instance
        self.transform_f = make_transfrom_f(include_torque=True)
        self.output_size = self.transform_f(self.env.get_current_state()).shape[0]
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
        self.output_size = self.transform_f(self.env.get_current_state()).shape[0] - 2
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
        if idx % 50 > 25:
            self.env.torque_random_set()
        self.append_result()

        state_list = torch.stack(list(self.cache_result))
        s0 = state_list[0]
        torques = get_torques(state_list[:-1])
        # endpoint diffs w.r.t the first endpoint
        diffs = get_arm1_end_points(state_list)[1:] - get_arm1_end_points(s0)[None, :]

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
        self.output_size = self.transform_f(self.env.get_current_state()).shape[0]
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