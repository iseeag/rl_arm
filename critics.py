from predictors import Predictor
from reward import reward
import torch

class Critic():

    def __init__(self, predictor, reward_f):
        self.predictor = predictor
        self.reward_f = reward_f

    def __call__(self, x):
        pred = self.predictor(x)
        reward = self.reward_f(x, pred)

        return reward

if __name__ == '__main__':
    predictor = Predictor(10, 2)
    critic = Critic(predictor, reward)
    critic(torch.zeros(10))

