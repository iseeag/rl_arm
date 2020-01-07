import torch

class Critic():

    def __init__(self, predictor, reward_f):
        self.predictor = predictor
        self.reward_f = reward_f

    def __call__(self, x: torch.Tensor, actions: torch.Tensor):
        pred_0, pred_a = self.predictor(x, actions) # predict both outcome of inaction and action
        rewards = self.reward_f(pred_0, pred_a)

        return rewards

if __name__ == '__main__':
    ...
    # predictor = Predictor(10, 2)
    # critic = Critic(predictor, reward_f)
    # q = critic(torch.zeros(10))

