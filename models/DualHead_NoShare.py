import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random, math, os, time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.adamw import AdamW
from utils.cyclic_scheduler import CyclicLRWithRestarts

from utils.early_stopping import EarlyStopping

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from Bias_Attention.utils.metrics import RMSLE

from utils.prepare_PM25 import test_pm25_single_station

# from visdom import Visdom
# from torchnet import meter
# from BlockAttention.multivariable.visual_loss import Visualizer

# from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########## Support


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def numpy_to_tvar(x):
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))


def plot_result(pred, true):
    pred_array = pred.data.numpy()
    true_array = true.data.numpy()

    plt.figure()
    plt.plot(pred_array, label='Predicted')
    plt.plot(true_array, label="True")
    plt.legend(loc='upper left')
    plt.pause(0.0001)


def show_attention(input_sentence, output_words, attentions):
    input_sentence = input_sentence.data.numpy()
    output_words = output_words.data.numpy()

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(input_sentence, rotation=90)
    ax.set_yticklabels(output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # show_plot_visdom()


def series_to_superviesed(x_timeseries,
                          y_timeseries,
                          n_memory_step,
                          n_forcast_step,
                          split=None):
    '''
        x_timeseries: input time series data, numpy array, (time_step, features)
        y_timeseries: target time series data,  numpy array, (time_step, features)
        n_memory_step: number of memory step in supervised learning, int
        n_forcast_step: number of forcase step in supervised learning, int
        split: portion of data to be used as train set, float, e.g. 0.8
    '''
    assert len(x_timeseries.shape
               ) == 2, 'x_timeseries must be shape of (time_step, features)'
    assert len(y_timeseries.shape
               ) == 2, 'y_timeseries must be shape of (time_step, features)'

    input_step, input_feature = x_timeseries.shape
    output_step, output_feature = y_timeseries.shape
    assert input_step == output_step, 'number of time_step of x_timeseries and y_timeseries are not consistent!'

    n_RNN_sample = input_step - n_forcast_step - n_memory_step + 1
    RNN_x = np.zeros((n_RNN_sample, n_memory_step, input_feature))
    RNN_y = np.zeros((n_RNN_sample, n_forcast_step, output_feature))

    for n in range(n_RNN_sample):
        RNN_x[n, :, :] = x_timeseries[n:n + n_memory_step, :]
        RNN_y[n, :, :] = y_timeseries[n + n_memory_step:n + n_memory_step +
                                      n_forcast_step, :]
    if split != None:
        assert (split <= 0.9) & (split >= 0.1), 'split not in reasonable range'
        return RNN_x[:int(split * len(RNN_x))], RNN_y[:int(split * len(RNN_x))], \
               RNN_x[int(split * len(RNN_x)) + 1:], RNN_y[int(split * len(RNN_x)) + 1:]
    else:
        return RNN_x, RNN_y, None, None


########### Dual Head Model
########### include left encoder, right encoder


class Shared_Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, enc_layers,
                 dec_layers, dropout_p):
        super(Shared_Encoder, self).__init__()

        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout_p = dropout_p

        self.input_linear = nn.Linear(self.input_dim, self.enc_hid_dim)
        self.gru_left = nn.GRU(input_size=self.enc_hid_dim,
                               hidden_size=self.enc_hid_dim,
                               num_layers=self.enc_layers,
                               bidirectional=True)
        self.gru_right = nn.GRU(input_size=self.enc_hid_dim,
                                hidden_size=self.enc_hid_dim,
                                num_layers=self.enc_layers,
                                bidirectional=True)
        self.output_linear_left = nn.Linear(self.enc_hid_dim * 2,
                                            self.dec_hid_dim)
        self.output_linear_right = nn.Linear(self.enc_hid_dim * 2,
                                             self.dec_hid_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input_before, input_after):

        # Left input
        embedded_before = self.dropout(
            torch.tanh(self.input_linear(input_before)))

        outputs_before, hidden_before = self.gru_left(embedded_before)

        hidden_before = torch.tanh(
            self.output_linear_left(
                torch.cat((hidden_before[-2, :, :], hidden_before[-1, :, :]),
                          dim=1)))

        embedded_after = self.dropout(
            torch.tanh(self.input_linear(input_after)))

        outputs_after, hidden_after = self.gru_right(embedded_after)

        hidden_after = torch.tanh(
            self.output_linear_right(
                torch.cat((hidden_after[-2, :, :], hidden_after[-1, :, :]),
                          dim=1)))

        # Only use hidden before to init decoder GRU
        # print('Init Hidden for Decoder:')
        hidden_decoder = hidden_before.repeat(self.dec_layers, 1, 1)

        # return outputs, hidden
        return outputs_before, outputs_after, hidden_decoder


class Cross_Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Cross_Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim,
                              self.dec_hid_dim)
        self.v = nn.Parameter(torch.rand(self.dec_hid_dim))

    def forward(self, hidden, encoder_outputs):

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # only pick up last layer hidden from decoder
        hidden = torch.unbind(hidden, dim=0)[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        energy = energy.permute(0, 2, 1)

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, dec_layers,
                 dropout_p, attention):
        super(Decoder, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dec_layers = dec_layers
        self.dropout_p = dropout_p
        self.attention = attention

        self.input_dec = nn.Linear(self.output_dim, self.dec_hid_dim)
        self.gru = nn.GRU(input_size=self.enc_hid_dim * 2 + self.dec_hid_dim,
                          hidden_size=self.dec_hid_dim,
                          num_layers=self.dec_layers)

        self.out = nn.Linear(
            self.enc_hid_dim * 2 + self.dec_hid_dim + self.dec_hid_dim,
            self.output_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden, encoder_outputs_left,
                encoder_outputs_right):

        input = input.unsqueeze(0)
        input = torch.unsqueeze(input, 2)

        embedded = self.dropout(torch.tanh(self.input_dec(input)))

        # concatenate two outputs

        encoder_outputs = torch.cat(
            (encoder_outputs_left, encoder_outputs_right), dim=0)

        a = self.attention(hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        gru_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.gru(gru_input, hidden)

        input_dec = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, input_dec), dim=1))

        return output.squeeze(1), hidden, a.squeeze(1)


class DualSSIM(nn.Module):
    def __init__(self, shared_encoder, decoder, device):
        super(DualSSIM, self).__init__()
        self.shared_encoder = shared_encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_left, src_right, trg, teacher_forcing_ratio=0.5):

        batch_size = src_left.shape[1]
        max_len = trg.shape[0]

        outputs = torch.zeros(max_len, batch_size,
                              self.decoder.output_dim).to(self.device)

        # save attn states
        decoder_attn = torch.zeros(max_len, batch_size, 20).to(self.device)

        # Shared Encoder
        encoder_outputs_left, encoder_outputs_right, hidden = self.shared_encoder(
            src_left, src_right)

        # only use y initial y
        output = src_left[-1, :, 0]

        for t in range(0, max_len):

            output, hidden, attn_weight = self.decoder(output, hidden,
                                                       encoder_outputs_left,
                                                       encoder_outputs_right)

            decoder_attn[t] = attn_weight.squeeze()

            outputs[t] = output.unsqueeze(1)

            teacher_force = random.random() < teacher_forcing_ratio

            output = (trg[t].view(-1) if teacher_force else output)

        return outputs, decoder_attn