from typing import *
import time
import numpy as np
from torch import optim
from functools import wraps
import torch.nn as nn
import torch

def get_nn_params(model, print_out=False):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    if print_out:
        print(f'number of parameters: {pp}')
    return pp

class Timer:
    """Simple timer.
    Example:
        >>> with Timer():
        >>>     ... # run whatever need to be timed
        time used: 4.56s # dummy responds
    """
    def __init__(self):
        self.time_stamp = None

    def __enter__(self):
        self.time_stamp = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'time used: {time.time() - self.time_stamp:.2f}s')

    def time_used(self):
        return f'{time.time() - self.time_stamp:.2f}s'


class AverageMeter:
    def __init__(self, max_len = 99):
        self.array = np.full([max_len], 1.0)
        self.max_len = max_len
        self.index = 0

    def log(self, v):
        if self.index == 0:
            self.array.fill(v)
        self.array[self.index % self.max_len] = v
        self.index += 1

    @property
    def value(self):
        return self.array.mean()

    @property
    def std(self):
        return self.array.std()

class OptimizerContext:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def __enter__(self):
        self.optimizer.zero_grad()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.optimizer.step()


def add_save_load_optimize_optimizer_optim_context(cls, instance):

    assert not hasattr(instance, 'save_path')
    assert not hasattr(instance, 'optimizer')

    instance.save_path = 'saved_models'
    instance.optimizer = optim.Adam(instance.parameters(), amsgrad=True)
    # instance.optimizer = optim.SGD(instance.parameters(), lr=1e-3, momentum=)

    instance.optim_cont = OptimizerContext(instance.optimizer)

    def save(self, name):
        torch.save(self.state_dict(), f'{self.save_path}/{name}')
        print(f'{self.save_path}/{name} saved')

    setattr(cls, 'save', save)


    def load(self, name):
        self.load_state_dict(torch.load(f'{self.save_path}/{name}'))
        self.eval()
        print('load successful')
        return True

    setattr(cls, 'load', load)


    def optimize_c(self):
        return self.optim_cont

    setattr(cls, 'optimize_c', optimize_c)

def add_auto_save(cls, instance, mode='min', n_delay=500):
    '''Example:
        >>> model = ...
        >>>model.set_auto_save_name('name').set_auto_save_delay(1000).toggle_auto_save()
        >>>'model save to path: path/name/pt'
        >>>loss = ...
        >>>model.save_by_score(loss)
        '''

    assert hasattr(instance, 'save_path')
    assert mode == 'min' or mode == 'max'
    assert not hasattr(instance, '_model_best_score')
    assert not hasattr(instance, '_auto_save_name')
    assert not hasattr(instance, '_auto_save_toggle')
    assert not hasattr(instance, '_auto_save_first_n_delay')

    instance._auto_save_toggle = False
    instance._auto_save_first_n_delay = n_delay
    if mode == 'min':
        instance._model_best_score = float('inf')
    else:
        instance._model_best_score = float('-inf')

    def save_by_score(self, score):
        if self._auto_save_first_n_delay > 0:
            self._auto_save_first_n_delay -= 1
            self._model_best_score = score
            return

        if self._auto_save_toggle is True:
            if mode == 'min':
                if score < self._model_best_score:
                    self._model_best_score = score
                    print(f'smallest score: {score} ', end='')
                    self.save(self._auto_save_name)
            elif mode == 'max':
                if score > self._model_best_score:
                    self._model_best_score = score
                    print(f'biggest score: {score} ', end='')
                    self.save(self._auto_save_name)

    def set_auto_save_name(self, name: str):
        self._auto_save_name = name
        print(f'set save name to {self._auto_save_name}')
        return self

    def set_auto_save_delay(self, n: int):
        self._auto_save_first_n_delay = n
        print(f'delay for {self._auto_save_first_n_delay} step')
        return self

    def toggle_auto_save(self):
        if self._auto_save_name == None and self._auto_save_toggle is False:
            raise Exception('Need a auto save name end with .pt! ')

        self._auto_save_toggle = not self._auto_save_toggle
        if self._auto_save_toggle:
            print(f'Auto saving is on! will save model to {self.save_path}/{self._auto_save_name}, start after '
                  f'{self._auto_save_first_n_delay} step if the given score to model.save_by_score(score) is the {mode}imum')
        else:
            print(f'Auto save is off')

        return self

    setattr(cls, 'save_by_score', save_by_score)
    setattr(cls, 'set_auto_save_name', set_auto_save_name)
    setattr(cls, 'set_auto_save_delay', set_auto_save_delay)
    setattr(cls, 'toggle_auto_save', toggle_auto_save)