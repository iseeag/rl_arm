import typing
from utils import get_nn_params, add_save_load_optimize_optimizer_optim_context
from interface import get_arm1_end_points, remove_torque, get_torques
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math



class PredictorLSTMTorque(nn.Module):
    '''
    insert environment state as hidden state and take up torque magnitudes as input
    generate a sequence of end_point_diffs (wrt last endpoint)
    Inputs: input, h_0
        input of shape: (batch, seq_len, torque_input_size)
        h_0 of shape: (batch, environment_state_size)
    Outputs: output
        output of shape: (batch, end_point_diff)
    '''
    def __init__(self, input_size, output_size, torque_input_size=2, expand_ratio=2, n_layer=2, base_rnn='gru'):
        super(PredictorLSTMTorque, self).__init__()
        self.cls = PredictorLSTMTorque
        self.input_size = input_size # 15 (env_state - torque)
        self.stepwise_input_size = torque_input_size
        self.expand_ratio = expand_ratio
        self.n_layer = n_layer
        self.base_rnn = base_rnn

        self.encoder0 = Encoder(input_size, compress_ratio=expand_ratio * n_layer) # for h_0
        if base_rnn == 'lstm':
            self.encoder1 = Encoder(input_size, compress_ratio=expand_ratio * n_layer) # for c_0
            self.rnn = nn.LSTM(torque_input_size, self.encoder0.out_size // n_layer, n_layer, batch_first=True)
        elif base_rnn == 'gru':
            self.rnn = nn.GRU(torque_input_size, self.encoder0.out_size // n_layer, n_layer, batch_first=True)
        elif base_rnn == 'rnn':
            self.rnn = nn.RNN(torque_input_size, self.encoder0.out_size // n_layer, n_layer, batch_first=True)
        else:
            raise(AttributeError('unknown base rnn in PredictorLSTM'))

        assert self.encoder0.out_size // n_layer == self.encoder0.out_size / n_layer  # sanity check
        self.decoder = Decoder(self.encoder0.out_size // n_layer, output_size)

        self.parameter_size = get_nn_params(self, True)
        add_save_load_optimize_optimizer_optim_context(self.cls, self)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, env_state, torques): # env_state, shape: (batch, seq, env_state)|torques, shape: (batch, seq, torques)

        h_size = self.rnn.hidden_size
        # h_0, shape need to be: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = self.encoder0(remove_torque(env_state))
        h_0 = torch.stack([h_0[:, i * h_size:(i + 1) * h_size] for i in range(self.rnn.num_layers)], dim=0)

        if self.base_rnn == 'lstm':
            # c_0, shape should be: (num_layers * num_directions, batch_size, hidden_size)
            c_0 = self.encoder1(remove_torque(env_state))
            c_0 = torch.stack([c_0[:, i * h_size:(i+1) * h_size] for i in range(self.rnn.num_layers)], dim=0)
            output, (h_n, c_n) = self.rnn(torques, (h_0, c_0))

        else: # for 'gru and rnn'
            output, h_n = self.rnn(torques, h_0)

        diffs = self.decoder(output)

        return diffs


class PredictorLSTM(nn.Module):

    def __init__(self, input_size, output_size, expand_ratio=3, seq_len=10, n_layer=2, base_rnn='lstm'):
        super(PredictorLSTM, self).__init__()
        self.input_size = input_size
        self.expand_ratio = expand_ratio
        self.seq_len = seq_len
        self.n_layer = n_layer
        self.base_rnn = base_rnn

        self.encoder0 = Encoder(input_size, compress_ratio=expand_ratio * n_layer) # for h_0
        if base_rnn == 'lstm':
            self.encoder1 = Encoder(input_size, compress_ratio=expand_ratio * n_layer) # for c_0
            self.rnn = nn.LSTM(1, self.encoder0.out_size // n_layer, n_layer, batch_first=True)
        elif base_rnn == 'gru':
            self.rnn = nn.GRU(1, self.encoder0.out_size // n_layer, n_layer, batch_first=True)
        elif base_rnn == 'rnn':
            self.rnn = nn.RNN(1, self.encoder0.out_size // n_layer, n_layer, batch_first=True)
        else:
            raise(AttributeError('unknown base rnn in PredictorLSTM'))

        assert self.encoder0.out_size // n_layer == self.encoder0.out_size / n_layer  # sanity check
        self.decoder = Decoder(self.encoder0.out_size // n_layer, output_size)

        self.parameter_size = get_nn_params(self, True)

        add_save_load_optimize_optimizer_optim_context(PredictorLSTM, self)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x): # x, shape (batch, input_size)
        batch_size, input_size = x.shape
        input = torch.zeros(batch_size, self.seq_len, 1) # shape: (batch_size, seq_len, input_size)

        h_size = self.rnn.hidden_size
        h_0 = self.encoder0(x)  # h_0, shape: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.stack([h_0[:, i * h_size:(i + 1) * h_size] for i in range(self.rnn.num_layers)], dim=0)

        if self.base_rnn == 'lstm':
            c_0 = self.encoder1(x) # c_0, shape: (num_layers * num_directions, batch_size, hidden_size)
            c_0 = torch.stack([c_0[:, i * h_size:(i+1) * h_size] for i in range(self.rnn.num_layers)], dim=0)
            output, (h_n, c_n) = self.rnn(input, (h_0, c_0))

        else: # for 'gru and rnn'
            output, h_n = self.rnn(input, h_0)

        x = self.decoder(output)

        return x


class Predictor(nn.Module):

    def __init__(self, input_size, output_size):
        super(Predictor, self).__init__()
        self.encoder = Encoder(input_size)
        self.stepper = Stepper(self.encoder.out_size)
        self.decoder = Decoder(self.stepper.out_size, output_size)
        self.parameter_size = get_nn_params(self, True)
        self.save_path = 'saved_models'

        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.stepper(x)
        x = self.decoder(x)
        return x

    def optimize(self, x, y):
        results = self(x)
        loss = self.criterion(results, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return results, loss.data

    def save(self, name):
        torch.save(self.state_dict(), f'{self.save_path}/{name}')

    def load(self, name):
        self.load_state_dict(torch.load(f'{self.save_path}/{name}'))
        self.eval()
        print('load successful')
        return True

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

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
        self.parameter_size = get_nn_params(self)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):
        for fc_name in self.fcs_list[:-1]: # skip relu for the last layer
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
        self.parameter_size = get_nn_params(self)

    def forward(self, x):
        for fc_name in self.fcs_list[:-1]: # skip relu for the last layer
            fc = getattr(self, fc_name)
            x = F.relu(fc(x))
        fc = getattr(self, self.fcs_list[-1])
        x = fc(x)
        return x

class Stepper(nn.Module):
    def __init__(self, in_size, out_size=None, expand_ratio=10, h_layer=1, recurrent_step=0):
        super(Stepper, self).__init__()
        self.recurrent_step = recurrent_step
        self.in_size = in_size
        self.out_size = in_size if out_size is None else out_size
        sizes = [in_size] + [math.floor(in_size * expand_ratio)] * h_layer + [self.out_size]
        self.sizes = [*zip(sizes[:-1], sizes[1:])]
        self.fcs_list = []
        # self.fc{i} = nn.Linear(in, out)
        for i, fc in enumerate([nn.Linear(in_size, out_size) for in_size, out_size in self.sizes]):
            setattr(self, f'fc{i}', fc)
            self.fcs_list.append(f'fc{i}')
        self.parameter_size = get_nn_params(self)

    def forward(self, x):
        for _ in range(self.recurrent_step + 1):
            for fc_name in self.fcs_list:
                fc = getattr(self, fc_name)
                x = F.relu(fc(x))
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
    predictor = Predictor(20, 20)
    pretrain_encoder_decoder(predictor.encoder, predictor.decoder)
