import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Predictor(nn.Module):

    def __init__(self):
        super(Predictor, self).__init__()
        self.encoder = Encoder(10)
        self.stepper = Stepper(self.encoder.out_size)
        self.decoder = Decoder(self.stepper.out_size, self.stepper.in_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_size):
        super(Encoder, self).__init__()
        self.in_size = in_size
        self.compress_ratio = 0.8
        self.mid_size = math.floor(self.in_size * self.compress_ratio)
        self.out_size = math.floor(self.mid_size * self.compress_ratio)
        self.fc0 = nn.Linear(self.in_size, self.mid_size)
        self.fc1 = nn.Linear(self.mid_size, self.out_size)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return x

class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.mid_size = math.floor((in_size + out_size)/2)
        self.fc0 = nn.Linear(self.in_size, self.mid_size)
        self.fc1 = nn.Linear(self.mid_size, self.out_size)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return x

class Stepper(nn.Module):
    def __init__(self, in_size, out_size=None, expand_ratio=1):
        super(Stepper, self).__init__()
        self.in_size = in_size
        self.out_size = in_size if out_size is None else out_size
        self.mid_size = math.floor(in_size * expand_ratio)
        self.fc0 = nn.Linear(self.in_size, self.mid_size)
        self.fc1 = nn.Linear(self.mid_size, self.out_size)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return x





