import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math

class Predictor(nn.Module):

    def __init__(self):
        super(Predictor, self).__init__()
        self.encoder = Encoder(20)
        self.stepper = Stepper(self.encoder.out_size)
        self.decoder = Decoder(self.stepper.out_size, self.encoder.in_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_size, compress_ratio=1.2):
        super(Encoder, self).__init__()
        self.in_size = in_size
        self.compress_ratio = compress_ratio
        self.mid_size = math.floor(self.in_size * self.compress_ratio)
        self.out_size = math.floor(self.mid_size * self.compress_ratio)
        self.fc0 = nn.Linear(self.in_size, self.mid_size)
        self.fc1 = nn.Linear(self.mid_size, self.out_size)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_size, out_size, compress_ratio=1.2):
        super(Decoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.compress_ratio = compress_ratio
        self.mid_size = math.floor(self.in_size * self.compress_ratio)
        self.fc0 = nn.Linear(self.in_size, self.mid_size)
        self.fc1 = nn.Linear(self.mid_size, self.out_size)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
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

def pretrain_encoder_decoder(encoder, decoder):
    class EncoderDecoder(nn.Module):
        def __init__(self, encoder, decoder):
            super(EncoderDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            x = encoder(x)
            x = decoder(x)
            return x

    encoder_decoder = EncoderDecoder(encoder, decoder)
    optimizer = optim.Adam(encoder_decoder.parameters())
    criterion = nn.MSELoss()
    for i in range(10000):
        train_set = torch.rand(1000,20)
        valid_set = torch.rand(1000,20)
        optimizer.zero_grad()
        loss = criterion(train_set, encoder_decoder(train_set))
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'epoch {i}: trn loss {loss.data:.6f}, '
                  f'vld loss {criterion(valid_set, encoder_decoder(valid_set)):.6f}')
    optimizer.zero_grad()

predictor = Predictor()
pretrain_encoder_decoder(predictor.encoder, predictor.decoder)