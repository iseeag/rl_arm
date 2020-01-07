#!/usr/bin/python
from utils import Timer, AverageMeter
from dataset import *
from predictors import Predictor, PredictorLSTM, PredictorLSTMTorque, PredictorOfTorque
from environment import EnvironmentState
from torch.utils.data import DataLoader
import torch


# -------------------------------- predictor training scripts --------------------------------------------------------
if __name__ == '__main__':
    step_list = [1, 8, 16, 24, 50, 100]
    print('training start now: ')
    for n_step in step_list:
        i = -1
        env = EnvironmentState()
        state_set = StateDataset(env, skip_step=n_step, size=7000000, random_torque=True, remove_torque=False)
        state_set_loader = DataLoader(state_set, batch_size=1024)

        predictor = Predictor(state_set.output_size, 2)
        predictor.stepper.recurrent_step = 0

        avm = AverageMeter()
        with Timer():
            for s in state_set_loader:
                i += 1
                d = s['s1'][:, 12:14] - s['s0'][:, 12:14] # with torques
                r, loss = predictor.optimize(s['s0'], d)
                avm.log(loss)
                if i % 10 == 0:
                    print(f'epoch {i}: trn loss {avm.value:.4f} {avm.std:.4f}, rmse {avm.value**0.5:.4f}, '
                          f'pred to stat error ratio: {torch.mean(torch.abs(r / d)):.4f} '
                          f'max_d {torch.max(d):.2f} {torch.min(d):.2f}, max_r {torch.max(r):.2f} {torch.min(r):.2f}')
        predictor.save(f'predictor{n_step}step.pt')

    print('training finished.')