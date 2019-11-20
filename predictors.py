import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math

class Predictor(nn.Module):

    def __init__(self, input_size):
        super(Predictor, self).__init__()
        self.encoder = Encoder(input_size)
        self.stepper = Stepper(self.encoder.out_size)
        self.decoder = Decoder(self.stepper.out_size, self.encoder.in_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.stepper(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_size, compress_ratio=1.2, n_layer=1):
        super(Encoder, self).__init__()
        self.in_size = in_size
        self.compress_ratio = compress_ratio
        sizes = [math.floor(in_size * compress_ratio**i) for i in range(n_layer + 1)]
        self.sizes = [*zip(sizes[:-1], sizes[1:])]
        self.out_size = sizes[-1]
        self.fcs_list = []
        for i, fc in enumerate([nn.Linear(in_size, out_size) for in_size, out_size in self.sizes]):
            setattr(self, f'fc{i}', fc)
            self.fcs_list.append(f'fc{i}')

    def forward(self, x):
        for fc_name in self.fcs_list[:-1]:
            fc = getattr(self, fc_name)
            x = F.relu(fc(x))
        fc = getattr(self, self.fcs_list[-1])
        x = fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_size, out_size, compress_ratio=1.2, n_layer=1):
        super(Decoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.compress_ratio = compress_ratio

        sizes = [math.floor(in_size + (out_size - in_size) * i / n_layer) for i in range(n_layer + 1)]
        assert(sizes[0] == in_size and sizes[-1] == out_size)
        self.sizes = [*zip(sizes[:-1], sizes[1:])]
        self.fcs_list = []
        for i, fc in enumerate([nn.Linear(in_size, out_size) for in_size, out_size in self.sizes]):
            setattr(self, f'fc{i}', fc)
            self.fcs_list.append(f'fc{i}')

    def forward(self, x):
        for fc_name in self.fcs_list[:-1]:
            fc = getattr(self, fc_name)
            x = F.relu(fc(x))
        fc = getattr(self, self.fcs_list[-1])
        x = fc(x)
        return x

class Stepper(nn.Module):
    def __init__(self, in_size, out_size=None, expand_ratio=2):
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

def pretrain_encoder_decoder(encoder, decoder, epochs_size=600):
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
    in_size = encoder_decoder.encoder.in_size
    for i in range(epochs_size):
        train_set = -1 + torch.rand(1000, in_size) * 2
        valid_set = -1 + torch.rand(1000, in_size) * 2
        optimizer.zero_grad()
        loss = criterion(train_set, encoder_decoder(train_set))
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'epoch {i}: trn loss {loss.data:.6f}, '
                  f'vld loss {criterion(valid_set, encoder_decoder(valid_set)):.6f}')
    optimizer.zero_grad()

if __name__ == '__main__':
    predictor = Predictor(15)
    pretrain_encoder_decoder(predictor.encoder, predictor.decoder)