import time
import numpy as np
import torch.optim.optimizer

def get_nn_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
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


class AverageMeter:
    def __init__(self, max_len = 80):
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
    def __init__(self, optimizer: torch.optim.Adam):
        self.optimizer = optimizer

    def __enter__(self):
        self.optimizer.zero_grad()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.optimizer.step()
