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

from BlockAttention.multivariable.standrisation import preprocess_df
from BlockAttention.multivariable.adamw import AdamW
from BlockAttention.multivariable.cyclic_scheduler import CyclicLRWithRestarts

from Bias_Attention.utils.early_stopping import EarlyStopping

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from Bias_Attention.utils.metrics import RMSLE

# from Dual-Head-SSIM.utils.prepare_PM25 import test_pm25_single_station


# from visdom import Visdom
# from torchnet import meter
# from BlockAttention.multivariable.visual_loss import Visualizer

from BlockAttention.multivariable.CBAM import BottlenNeck

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

    # print('here')
    # print(attentions.data.numpy())

    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(input_sentence, rotation=90)
    ax.set_yticklabels(output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # show_plot_visdom()


def series_to_superviesed(x_timeseries, y_timeseries, n_memory_step, n_forcast_step, split=None):
    '''
        x_timeseries: input time series data, numpy array, (time_step, features)
        y_timeseries: target time series data,  numpy array, (time_step, features)
        n_memory_step: number of memory step in supervised learning, int
        n_forcast_step: number of forcase step in supervised learning, int
        split: portion of data to be used as train set, float, e.g. 0.8
    '''
    assert len(x_timeseries.shape) == 2, 'x_timeseries must be shape of (time_step, features)'
    assert len(y_timeseries.shape) == 2, 'y_timeseries must be shape of (time_step, features)'

    input_step, input_feature = x_timeseries.shape
    output_step, output_feature = y_timeseries.shape
    assert input_step == output_step, 'number of time_step of x_timeseries and y_timeseries are not consistent!'

    n_RNN_sample = input_step - n_forcast_step - n_memory_step + 1
    RNN_x = np.zeros((n_RNN_sample, n_memory_step, input_feature))
    RNN_y = np.zeros((n_RNN_sample, n_forcast_step, output_feature))

    for n in range(n_RNN_sample):
        RNN_x[n, :, :] = x_timeseries[n:n + n_memory_step, :]
        RNN_y[n, :, :] = y_timeseries[n + n_memory_step:n + n_memory_step + n_forcast_step, :]
    if split != None:
        assert (split <= 0.9) & (split >= 0.1), 'split not in reasonable range'
        return RNN_x[:int(split * len(RNN_x))], RNN_y[:int(split * len(RNN_x))], \
               RNN_x[int(split * len(RNN_x)) + 1:], RNN_y[int(split * len(RNN_x)) + 1:]
    else:
        return RNN_x, RNN_y, None, None


########### Dual Head Model
########### include left encoder, right encoder


# class DualEncoder(nn.Module):
#     def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, enc_layers, dec_layers, dropout_p):
#         super(DualEncoder, self).__init__()
#
#         self.input_dim = input_dim
#         self.enc_hid_dim = enc_hid_dim
#         self.dec_hid_dim = dec_hid_dim
#         self.enc_layers = enc_layers
#         self.dec_layers = dec_layers
#         self.dropout_p = dropout_p
#
#         self.input_linear = nn.Linear(self.input_dim, self.enc_hid_dim)
#         self.gru = nn.GRU(input_size=self.enc_hid_dim,hidden_size=self.enc_hid_dim,num_layers=self.enc_layers,bidirectional=True)
#         self.output_linear = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)
#         self.dropout = nn.Dropout(self.dropout_p)
#
#         def forward(self, left_input, right_input):
#             # print('input')
#             # print(input.size())
#
#
#             # for left input
#             left_embedded = self.dropout(torch.tanh(self.input_linear(left_input)))
#
#             # print('Embedded')
#             # print(embedded.size())
#
#             left_outputs, left_hidden = self.gru(left_embedded)
#
#             # print('Encoder')
#             #
#             # print(outputs)
#             # print(hidden)
#             # print(cell)
#
#             left_hidden = torch.tanh(self.output_linear(torch.cat((left_hidden[-2, :, :], left_hidden[-1, :, :]), dim=1)))
#
#             # for different number of decoder layers
#             left_hidden = left_hidden.repeat(self.dec_layers, 1, 1)
#
#
#             # for right input
#             right_embedded = self.dropout(torch.tanh(self.input_linear(right_input)))
#
#             # print('Embedded')
#             # print(embedded.size())
#
#             right_outputs, right_hidden = self.gru(right_embedded)
#
#             # print('Encoder')
#             #
#             # print(outputs)
#             # print(hidden)
#             # print(cell)
#
#             right_hidden = torch.tanh(self.output_linear(torch.cat((right_hidden[-2, :, :], right_hidden[-1, :, :]), dim=1)))
#
#             # for different number of decoder layers
#             right_hidden = right_hidden.repeat(self.dec_layers, 1, 1)
#
#
#             return outputs, (hidden, hidden)





class Shared_Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, enc_layers, dec_layers, dropout_p):
        super(Shared_Encoder, self).__init__()

        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout_p = dropout_p

        self.input_linear = nn.Linear(self.input_dim, self.enc_hid_dim)
        self.gru = nn.GRU(input_size=self.enc_hid_dim,hidden_size=self.enc_hid_dim,num_layers=self.enc_layers,bidirectional=True)
        self.output_linear = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input_before, input_after):
        # print('input')
        # print(input.size())

        # Left input
        embedded_before = self.dropout(torch.tanh(self.input_linear(input_before)))

        # print('Embedded')
        # print(embedded.size())

        outputs_before, hidden_before = self.gru(embedded_before)

        # print('Encoder')
        #
        # print(outputs)
        # print(hidden)
        # print(cell)

        hidden_before = torch.tanh(self.output_linear(torch.cat((hidden_before[-2, :, :], hidden_before[-1, :, :]), dim=1)))




        # # for different number of decoder layers
        # hidden = hidden.repeat(self.dec_layers, 1, 1)



        # Right input

        embedded_after = self.dropout(torch.tanh(self.input_linear(input_after)))

        # print('Embedded')
        # print(embedded.size())

        outputs_after, hidden_after = self.gru(embedded_after)

        # print('Encoder')
        #
        # print(outputs)
        # print(hidden)
        # print(cell)

        hidden_after = torch.tanh(self.output_linear(torch.cat((hidden_after[-2, :, :], hidden_after[-1, :, :]), dim=1)))

        # # for different number of decoder layers
        # hidden = hidden.repeat(self.dec_layers, 1, 1)

        # return outputs, hidden
        return outputs_before, hidden_before, outputs_after, hidden_after








# class Left_Encoder(nn.Module):
#     def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, enc_layers, dec_layers, dropout_p):
#         super(Left_Encoder, self).__init__()
#
#         self.input_dim = input_dim
#         self.enc_hid_dim = enc_hid_dim
#         self.dec_hid_dim = dec_hid_dim
#         self.enc_layers = enc_layers
#         self.dec_layers = dec_layers
#         self.dropout_p = dropout_p
#
#         self.input_linear = nn.Linear(self.input_dim, self.enc_hid_dim)
#         self.gru = nn.GRU(input_size=self.enc_hid_dim,hidden_size=self.enc_hid_dim,num_layers=self.enc_layers,bidirectional=True)
#         self.output_linear = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)
#         self.dropout = nn.Dropout(self.dropout_p)
#
#     def forward(self, input):
#         # print('input')
#         # print(input.size())
#
#         embedded = self.dropout(torch.tanh(self.input_linear(input)))
#
#         # print('Embedded')
#         # print(embedded.size())
#
#         outputs, hidden = self.gru(embedded)
#
#         # print('Encoder')
#         #
#         # print(outputs)
#         # print(hidden)
#         # print(cell)
#
#         hidden = torch.tanh(self.output_linear(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
#
#         # for different number of decoder layers
#         hidden = hidden.repeat(self.dec_layers, 1, 1)
#
#         return outputs, (hidden, hidden)
#
#
#
#
# class Right_Encoder(nn.Module):
#     def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, enc_layers, dec_layers, dropout_p):
#         super(Right_Encoder, self).__init__()
#
#         self.input_dim = input_dim
#         self.enc_hid_dim = enc_hid_dim
#         self.dec_hid_dim = dec_hid_dim
#         self.enc_layers = enc_layers
#         self.dec_layers = dec_layers
#         self.dropout_p = dropout_p
#
#         self.input_linear = nn.Linear(self.input_dim, self.enc_hid_dim)
#         # self.lstm = nn.LSTM(input_size=self.enc_hid_dim, hidden_size=self.enc_hid_dim, num_layers=self.enc_layers,
#         #                     bidirectional=True)
#         self.gru = nn.GRU(input_size=self.enc_hid_dim,hidden_size=self.enc_hid_dim,num_layers=self.enc_layers,bidirectional=True)
#         self.output_linear = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)
#         self.dropout = nn.Dropout(self.dropout_p)
#
#     def forward(self, input):
#         # print('input')
#         # print(input.size())
#
#         embedded = self.dropout(torch.tanh(self.input_linear(input)))
#
#         # print('Embedded')
#         # print(embedded.size())
#
#         outputs, hidden = self.gru(embedded)
#
#         # print('Encoder')
#         #
#         # print(outputs)
#         # print(hidden)
#         # print(cell)
#
#         hidden = torch.tanh(self.output_linear(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
#
#         # for different number of decoder layers
#         hidden = hidden.repeat(self.dec_layers, 1, 1)
#
#         return outputs, (hidden, hidden)


class Cross_Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Cross_Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim, self.dec_hid_dim)
        self.v = nn.Parameter(torch.rand(self.dec_hid_dim))

    def forward(self, hidden, encoder_outputs_left, encoder_outputs_right):


        batch_size = encoder_outputs_left.shape[1]
        src_len_left = encoder_outputs_left.shape[0]
        src_len_right = encoder_outputs_right.shape[0]
        src_len = src_len_left+src_len_right

        # print(hidden.size())


        # concatenate two outputs




        # only pick up last layer hidden from decoder
        hidden = torch.unbind(hidden, dim=0)[0]
        # hidden = hidden[-1, :, :].squeeze(0)

        # print(hidden.size())

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # print(hidden.size())


        # encoder output
        encoder_outputs_left = encoder_outputs_left.permute(1, 0, 2)
        # print(encoder_outputs_left.size())
        encoder_outputs_right = encoder_outputs_right.permute(1, 0, 2)
        # print(encoder_outputs_right.size())



        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # print(hidden.size())
        # print(encoder_outputs.size())
        # print('-----------------')

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        energy = energy.permute(0, 2, 1)

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)



        return F.softmax(attention, dim=1)













# class Encoder(nn.Module):
#     def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, enc_layers, dec_layers, dropout_p):
#         super(Encoder, self).__init__()
#
#         self.input_dim = input_dim
#         self.enc_hid_dim = enc_hid_dim
#         self.dec_hid_dim = dec_hid_dim
#         self.enc_layers = enc_layers
#         self.dec_layers = dec_layers
#         self.dropout_p = dropout_p
#
#         self.input_linear = nn.Linear(self.input_dim, self.enc_hid_dim)
#         self.lstm = nn.LSTM(input_size=self.enc_hid_dim, hidden_size=self.enc_hid_dim, num_layers=self.enc_layers,
#                             bidirectional=True)
#         self.output_linear = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)
#         self.dropout = nn.Dropout(self.dropout_p)
#
#     def forward(self, input):
#         # print('input')
#         # print(input.size())
#
#         embedded = self.dropout(torch.tanh(self.input_linear(input)))
#
#         # print('Embedded')
#         # print(embedded.size())
#
#         outputs, (hidden, cell) = self.lstm(embedded)
#
#         # print('Encoder')
#         #
#         # print(outputs)
#         # print(hidden)
#         # print(cell)
#
#         hidden = torch.tanh(self.output_linear(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
#
#         # for different number of decoder layers
#         hidden = hidden.repeat(self.dec_layers, 1, 1)
#
#         return outputs, (hidden, hidden)


# class Attention(nn.Module):
#     def __init__(self, enc_hid_dim, dec_hid_dim, output_dim):
#         super(Attention, self).__init__()
#
#         self.enc_hid_dim = enc_hid_dim
#         self.dec_hid_dim = dec_hid_dim
#         self.output_dim = output_dim
#
#         self.attn = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim, self.dec_hid_dim)
#         self.v = nn.Parameter(torch.rand(self.dec_hid_dim))
#
#         # self.emb_output = nn.Linear(self.output_dim, self.dec_hid_dim)
#         self.bias_attention = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim, self.dec_hid_dim)
#         self.v2 = nn.Parameter(torch.rand(self.dec_hid_dim))
#
#         self.posneg = nn.Linear(2, 1)
#
#     def forward(self, hidden, encoder_outputs, output, output_true):
#         batch_size = encoder_outputs.shape[1]
#         src_len = encoder_outputs.shape[0]
#
#         # only pick up last layer hidden
#         hidden = torch.unbind(hidden, dim=0)[0]
#
#         hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
#
#         encoder_outputs = encoder_outputs.permute(1, 0, 2)
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
#         energy = energy.permute(0, 2, 1)
#
#         v = self.v.repeat(batch_size, 1).unsqueeze(1)
#
#         main_attention = torch.bmm(v, energy)
#         main_attention = main_attention.permute(0, 2, 1)
#
#         # bias attention
#
#         # calculate bias
#
#         # output = output.unsqueeze(1)
#
#         # print('Bias_attention:')
#         # print('output:{}'.format(output.size()))
#         # print('ground_truth:{}'.format(output_true.size()))
#
#         bias = output - output_true
#
#         # print('bias:{}'.format(bias.size()))
#         #
#         # bias = self.emb_output(bias)
#
#         # print('bias2:{}'.format(bias.size()))
#
#         bias = bias.unsqueeze(1).repeat(1, src_len, 1)
#
#         # print('bias3:{}'.format(bias.size()))
#
#         bias_energy = torch.tanh(self.bias_attention(torch.cat((bias, encoder_outputs), dim=2)))
#
#         bias_energy = bias_energy.permute(0, 2, 1)
#
#         v2 = self.v2.repeat(batch_size, 1).unsqueeze(1)
#
#         bias_attention = torch.bmm(v2, bias_energy)
#
#         bias_attention = bias_attention.permute(0, 2, 1)
#
#         # print('bias_attention:{}'.format(bias_attention.size()))
#
#         final_attention = torch.tanh(self.posneg(torch.cat((main_attention, bias_attention), dim=2)))
#
#         final_attention = final_attention.squeeze(2)
#
#         return F.softmax(final_attention, dim=1)





# class Global_Attention(nn.Module):
#     def __init__(self, enc_hid_dim, dec_hid_dim):
#         super(Global_Attention, self).__init__()
#
#         self.enc_hid_dim = enc_hid_dim
#         self.dec_hid_dim = dec_hid_dim
#
#         self.attn = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim, self.dec_hid_dim)
#         self.v = nn.Parameter(torch.rand(self.dec_hid_dim))
#
#     def forward(self, hidden, encoder_outputs):
#         batch_size = encoder_outputs.shape[1]
#         src_len = encoder_outputs.shape[0]
#
#         # print(hidden.size())
#
#         # only pick up last layer hidden
#         hidden = torch.unbind(hidden, dim=0)[0]
#         # hidden = hidden[-1, :, :].squeeze(0)
#
#         # print(hidden.size())
#
#         hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
#
#         # print(hidden.size())
#
#         encoder_outputs = encoder_outputs.permute(1, 0, 2)
#
#         # print(hidden.size())
#         # print(encoder_outputs.size())
#         # print('-----------------')
#
#         energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
#
#         energy = energy.permute(0, 2, 1)
#
#         v = self.v.repeat(batch_size, 1).unsqueeze(1)
#
#         attention = torch.bmm(v, energy).squeeze(1)
#
#
#
#         return F.softmax(attention, dim=1)



class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, dec_layers, dropout_p, attention):
        super(Decoder, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dec_layers = dec_layers
        self.dropout_p = dropout_p
        self.attention = attention

        # self.input_dec = nn.Linear(output_dim, enc_hid_dim)
        self.input_dec = nn.Linear(self.output_dim, self.dec_hid_dim)
        self.lstm = nn.LSTM(input_size=self.enc_hid_dim * 2 + self.dec_hid_dim, hidden_size=self.dec_hid_dim,
                            num_layers=self.dec_layers)
        self.out = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim + self.dec_hid_dim, self.output_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden, cell, encoder_outputs):
        # print('Decoder:')
        # print('input:{}'.format(input.size()))

        input = input.unsqueeze(0)
        input = torch.unsqueeze(input, 2)

        embedded = self.dropout(torch.tanh(self.input_dec(input)))

        # print('embedded:{}'.format(embedded))

        # print('Decoder:')
        # print('embedded:{}'.format(embedded.size()))

        # # only pick up last layer hidden
        # hidden = hidden[-1,:,:].squeeze(0)

        a = self.attention(hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        lstm_input = torch.cat((embedded, weighted), dim=2)

        # print('lstm_input:{}'.format(lstm_input.size()))

        # hidden_2 = hidden.expand(self.dec_layers,-1,-1)

        # h_t, c_t = hidden[0][-2:], hidden[1][-2:]
        # decoder_hidden = torch.cat((h_t[0].unsqueeze(0), h_t[1].unsqueeze(0)), 2), torch.cat(
        #     (c_t[0].unsqueeze(0), c_t[1].unsqueeze(0)), 2)

        # # for different number of decoder layers
        # hidden = hidden.repeat(self.dec_layers,1,1)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # assert (output == hidden).all()

        input_dec = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        # print('input_dec:{}'.format(input_dec))
        # print('output:{}'.format(output))
        # print('weighted:{}'.format(weighted))

        # output = F.softplus(self.out(torch.cat((output, weighted, input_dec), dim=1)))
        output = self.out(torch.cat((output, weighted, input_dec), dim=1))

        # print('output:{}'.format(output))

        return output.squeeze(1), (hidden, cell), a






class DualSSIM(nn.Module):
    def __init__(self, shared_encoder, decoder, device):
        super(DualSSIM, self).__init__()
        self.shared_encoder = shared_encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_left, src_right, trg, teacher_forcing_ratio=0.5):
        # print(src.size())
        # print(trg)

        batch_size = src_left.shape[1]
        max_len = trg.shape[0]

        outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim).to(self.device)

        # save attn states

        decoder_attn = torch.zeros(max_len, src_left.shape[0]).to(self.device)

        # print(outputs.size())

        # print(decoder_attn.size())

        # print('0')
        # print(src)

        # left input
        encoder_outputs_left, hidden_left = self.shared_encoder(src_left,src_right)

        # # right input
        # encoder_outputs_right, hidden_right = self.Shared_Encoder(src_right)

        # only use y initial y
        output = src_left[-1, :, 0]

        # print('1')
        # # print(output.size())
        # print(encoder_outputs)
        # print(hidden)
        # print(cell)
        # print(output)

        for t in range(0, max_len):
            # print('output {} at {}'.format(output,t))

            # output, (hidden, cell), attn_weight = self.decoder(output, hidden, cell, encoder_outputs, trg[t])
            output, hidden, attn_weight = self.decoder(output, hidden, encoder_outputs_left, encoder_outputs_right)

            # print('2')
            # print(output.size())
            # print(encoder_outputs)
            # print(hidden)
            # print(cell)
            # print(attn_weight.size())

            # decoder_attn[t] = attn_weight.squeeze()

            # print(attn_weight.numpy())

            outputs[t] = output.unsqueeze(1)

            teacher_force = random.random() < teacher_forcing_ratio

            output = (trg[t].view(-1) if teacher_force else output)

            # output = output.squeeze()

            # print('2')
            # print(output.size())
            # print(output)
            #
            # print('3')
            # print(trg[t].size())
            # print(trg[t])

        # return outputs, decoder_attn
        return outputs






# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder, device):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device
#
#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         # print(src.size())
#         # print(trg)
#
#         batch_size = src.shape[1]
#         max_len = trg.shape[0]
#
#         outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim).to(self.device)
#
#         # save attn states
#
#         decoder_attn = torch.zeros(max_len, src.shape[0]).to(self.device)
#
#         # print(outputs.size())
#
#         # print(decoder_attn.size())
#
#         # print('0')
#         # print(src)
#
#         encoder_outputs, (hidden, cell) = self.encoder(src)
#
#         # only use y initial y
#         output = src[-1, :, 0]
#
#         # print('1')
#         # # print(output.size())
#         # print(encoder_outputs)
#         # print(hidden)
#         # print(cell)
#         # print(output)
#
#         for t in range(0, max_len):
#             # print('output {} at {}'.format(output,t))
#
#             # output, (hidden, cell), attn_weight = self.decoder(output, hidden, cell, encoder_outputs, trg[t])
#             output, (hidden, cell), attn_weight = self.decoder(output, hidden, cell, encoder_outputs)
#
#             # print('2')
#             # print(output.size())
#             # print(encoder_outputs)
#             # print(hidden)
#             # print(cell)
#             # print(attn_weight.size())
#
#             # decoder_attn[t] = attn_weight.squeeze()
#
#             # print(attn_weight.numpy())
#
#             outputs[t] = output.unsqueeze(1)
#
#             teacher_force = random.random() < teacher_forcing_ratio
#
#             output = (trg[t].view(-1) if teacher_force else output)
#
#             # output = output.squeeze()
#
#             # print('2')
#             # print(output.size())
#             # print(output)
#             #
#             # print('3')
#             # print(trg[t].size())
#             # print(trg[t])
#
#         # return outputs, decoder_attn
#         return outputs


def train(model, optimizer, criterion, X_train, y_train):
    # model.train()

    iter_per_epoch = int(np.ceil(X_train.shape[0] * 1. / BATCH_SIZE))
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)

    n_iter = 0

    perm_idx = np.random.permutation(X_train.shape[0])

    # train for each batch

    for t_i in range(0, X_train.shape[0], BATCH_SIZE):
        batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

        x_train_batch = np.take(X_train, batch_idx, axis=0)
        y_train_batch = np.take(y_train, batch_idx, axis=0)

        loss = train_iteration(model, optimizer, criterion, CLIP, WD, x_train_batch, y_train_batch)

        # if t_i % 50 == 0:
        #     print('batch_loss:{}'.format(loss))

        iter_losses[t_i // BATCH_SIZE] = loss

        # writer.add_scalars('Train_loss', {'train_loss': iter_losses[t_i // BATCH_SIZE]},
        #                    n_iter)

        # if (j / t_cfg.batch_size) % 50 == 0:
        #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / t_cfg.batch_size, loss)
        n_iter += 1

    return np.mean(iter_losses[range(0, iter_per_epoch)])


def train_iteration(model, optimizer, criterion, clip, wd, X_train, y_train):
    model.train()
    optimizer.zero_grad()

    X_train = np.transpose(X_train, [1, 0, 2])
    y_train = np.transpose(y_train, [1, 0, 2])

    # print('X_train:{}'.format(X_train))
    # print('y_train:{}'.format(y_train))

    X_train_tensor = numpy_to_tvar(X_train)
    y_train_tensor = numpy_to_tvar(y_train)

    # print(y_train_tensor.size())

    # output, _ = model(X_train_tensor, y_train_tensor, 0)
    output = model(X_train_tensor, y_train_tensor)

    # trg = [trg sent len, batch size]
    # output = [trg sent len, batch size, output dim]

    output = output.view(-1)

    # print(output)

    y_train_tensor = y_train_tensor.view(-1)

    # print('output:{}'.format(output))
    # print('y_train_tensor:{}'.format(y_train_tensor))

    # print('3')
    # print(output)
    # print(y_train_tensor)

    # trg = [(trg sent len - 1) * batch size]
    # output = [(trg sent len - 1) * batch size, output dim]

    loss = criterion(output, y_train_tensor)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    # # the block below changes weight decay in adam
    # for group in optimizer.param_groups:
    #     for param in group['params']:
    #         param.data = param.data.add(-wd * group['lr'], param.data)

    optimizer.step()

    scheduler.batch_step()

    # loss_meter.add(loss.item())

    return loss.item()


### evaluate

def evaluate(model, criterion, X_test, y_test):
    # model.eval()

    epoch_loss = 0
    iter_per_epoch = int(np.ceil(X_test.shape[0] * 1. / BATCH_SIZE))
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)
    # other loss: MAE RMSLE
    iter_multiloss = [np.zeros(EPOCHS * iter_per_epoch), np.zeros(EPOCHS * iter_per_epoch),np.zeros(EPOCHS * iter_per_epoch)]
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)
    perm_idx = np.random.permutation(X_test.shape[0])

    n_iter = 0

    with torch.no_grad():
        for t_i in range(0, X_test.shape[0], BATCH_SIZE):
            batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

            x_test_batch = np.take(X_test, batch_idx, axis=0)
            y_test_batch = np.take(y_test, batch_idx, axis=0)

            loss, mae, rmsle, rmse = evaluate_iteration(model, criterion, x_test_batch, y_test_batch)
            iter_losses[t_i // BATCH_SIZE] = loss
            iter_multiloss[0][t_i // BATCH_SIZE] = mae
            iter_multiloss[1][t_i // BATCH_SIZE] = rmsle
            iter_multiloss[2][t_i // BATCH_SIZE] = rmse

            # writer.add_scalars('Val_loss', {'val_loss': iter_losses[t_i // BATCH_SIZE]},
            #                    n_iter)

            n_iter += 1

    return np.mean(iter_losses[range(0, iter_per_epoch)]), np.mean(iter_multiloss[0][range(0, iter_per_epoch)]), np.mean(
        iter_multiloss[1][range(0, iter_per_epoch)]), np.mean(iter_multiloss[2][range(0, iter_per_epoch)])


def evaluate_iteration(model, criterion, x_test, y_test):
    model.eval()

    x_test = np.transpose(x_test, [1, 0, 2])
    y_test = np.transpose(y_test, [1, 0, 2])

    x_test_tensor = numpy_to_tvar(x_test)
    y_test_tensor = numpy_to_tvar(y_test)

    # output, decoder_attn = model(x_test_tensor, y_test_tensor, 0)
    output = model(x_test_tensor, y_test_tensor, 0)

    # trg = [trg sent len, batch size]
    # output = [trg sent len, batch size, output dim]

    output = output.view(-1)
    y_test_tensor = y_test_tensor.view(-1)

    # print('4')
    # print(output)
    # print(y_test_tensor)
    # print(x_test_tensor)

    # trg = [(trg sent len - 1) * batch size]
    # output = [(trg sent len - 1) * batch size, output dim]

    loss = criterion(output, y_test_tensor)

    # metric
    output_numpy = output.cpu().data.numpy()
    y_test_numpy = y_test_tensor.cpu().data.numpy()

    output_numpy = scaler_y.inverse_transform(output_numpy)
    y_test_numpy = scaler_y.inverse_transform(y_test_numpy)

    loss_mae = mean_absolute_error(y_test_numpy,output_numpy)
    loss_RMSLE = RMSLE(y_test_numpy,output_numpy)
    loss_RMSE = np.sqrt(mean_squared_error(y_test_numpy,output_numpy))




    # test_loss_meter.add(loss.item())

    # plot_result(output, y_test_tensor)
    # show_attention(x_test_tensor,output,decoder_attn)

    return loss.item(), loss_mae, loss_RMSLE, loss_RMSE


if __name__ == "__main__":

    INPUT_DIM = 54
    OUTPUT_DIM = 1
    ENC_HID_DIM = 20
    DEC_HID_DIM = 20
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    ECN_Layers = 2
    DEC_Layers = 2
    LR = 0.001  # learning rate
    WD = 0.1  # weight decay
    CLIP = 1
    EPOCHS = 50
    BATCH_SIZE = 100

    # Data
    n_memory_steps = 20  # length of input
    n_forcast_steps = 10  # length of output
    train_test_split = 0.8  # protion as train set
    validation_split = 0.2  # protion as validation set
    test_size = 0.1
    ## Data Processing

    filepath = r'C:\Users\ZHA244\Coding\Pytorch_based\BlockAttention\multivariable\newPM.csv'
    df = pd.read_csv(filepath, dayfirst=True)

    X_train, X_test, y_train, y_test, scaler_x, scaler_y = preprocess_df(df, n_memory_steps, n_forcast_steps, test_size,
                                                                         SEED)

    print('\nsize of x_train, y_train, x_test, y_test:')
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_train = X_train[:39400, :, :]
    y_train = y_train[:39400, :, :]
    #
    #
    X_test = X_test[:4350, :, :]
    y_test = y_test[:4350, :, :]



    print('split train/test array')
    X_test_list = np.split(X_test, [5,10] ,axis=1)
    X_train_list = np.split(X_train, [5,10] ,axis=1)





    # time series to image










    # Model


    cross_attn = Cross_Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Shared_Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ECN_Layers, DEC_Layers, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_Layers, DEC_DROPOUT, cross_attn)

    model = DualSSIM(enc, dec, device).to(device)
    model.apply(init_weights)

    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # AdamW
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = CyclicLRWithRestarts(optimizer, BATCH_SIZE, 39400, restart_period=5, t_mult=1.2, policy="cosine")


    criterion = nn.MSELoss()

    # # tensorboardX
    # writer = SummaryWriter('./log/exp-1')

    # dummy_input = torch.zeros(10,100, 1).to(device)
    # dummy_output = torch.zeros(10,100, 1).to(device)
    #
    #
    # with SummaryWriter(comment='Seq2Seq') as w:
    #     w.add_graph(model, (dummy_input, dummy_output), verbose=False)


    # # visulization visdom
    # vis = Visualizer(env='attention')
    #
    # loss_meter = meter.AverageValueMeter()
    # test_loss_meter = meter.AverageValueMeter()



    # Early Stopping
    # initialize the early_stopping object
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 20
    early_stopping = EarlyStopping(patience=patience, verbose=True)


    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):

        train_epoch_losses = np.zeros(EPOCHS)
        evaluate_epoch_losses = np.zeros(EPOCHS)

        # loss_meter.reset()

        # print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        print('Epoch:', epoch)

        scheduler.step()

        start_time = time.time()
        train_loss = train(model, optimizer, criterion, X_train, y_train)
        valid_loss,_,_,_ = evaluate(model, criterion, X_test, y_test, scaler_x, scaler_y)
        end_time = time.time()
        #
        # scheduler.step()

        # # visulization
        # vis.plot_many_stack({'train_loss': loss_meter.value()[0],'test_loss': test_loss_meter.value()[0]})

        train_epoch_losses[epoch] = train_loss
        evaluate_epoch_losses[epoch] = valid_loss

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break




        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), 'HPC-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # prediction

    #
    # model.load_state_dict(torch.load('checkpoint.pt',map_location='cpu'))
    #
    # test_loss, test_mae, test_rmsle, test_rmse = evaluate(model, criterion, X_test, y_test, scaler_x, scaler_y)
    #
    # # plt.show()
    #
    # print(f'| Test Loss: {test_loss:.4f} | Test PPL: {math.exp(test_loss):7.4f} |')
    # print(f'| MAE: {test_mae:.4f} | Test PPL: {math.exp(test_mae):7.4f} |')
    # print(f'| RMSLE: {test_rmsle:.4f} | Test PPL: {math.exp(test_rmsle):7.4f} |')
    # print(f'| RMSE: {test_rmse:.4f} | Test PPL: {math.exp(test_rmse):7.4f} |')
