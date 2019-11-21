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
        self.gru = nn.GRU(input_size=self.enc_hid_dim,
                          hidden_size=self.enc_hid_dim,
                          num_layers=self.enc_layers,
                          bidirectional=True)
        self.output_linear = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input_before, input_after):
        print('Left input')
        print(input_before.size())

        # Left input
        embedded_before = self.dropout(
            torch.tanh(self.input_linear(input_before)))

        print('Embedded Left')
        print(embedded_before.size())

        outputs_before, hidden_before = self.gru(embedded_before)

        print('Encoder Left')

        print('outputs_before:{}'.format(outputs_before.size()))
        print('hidden_before:{}'.format(hidden_before.size()))

        hidden_before = torch.tanh(
            self.output_linear(
                torch.cat((hidden_before[-2, :, :], hidden_before[-1, :, :]),
                          dim=1)))

        print('Hidden Left')

        print('hidden_before:{}'.format(hidden_before.size()))

        print('--------------')

        # # for different number of decoder layers
        # hidden = hidden.repeat(self.dec_layers, 1, 1)

        print('Right input')
        print(input_after.size())

        embedded_after = self.dropout(
            torch.tanh(self.input_linear(input_after)))

        print('Embedded Right')
        print(embedded_after.size())

        outputs_after, hidden_after = self.gru(embedded_after)

        print('Encoder Right')

        print('outputs_after:{}'.format(outputs_after.size()))
        print('hidden_after:{}'.format(hidden_after.size()))

        hidden_after = torch.tanh(
            self.output_linear(
                torch.cat((hidden_after[-2, :, :], hidden_after[-1, :, :]),
                          dim=1)))

        print('Hidden Right')

        print('hidden_after:{}'.format(hidden_after.size()))

        print('--------------2')
        # # for different number of decoder layers
        # hidden = hidden.repeat(self.dec_layers, 1, 1)

        # Only use hidden before to init decoder GRU
        print('Init Hidden for Decoder:')
        hidden_decoder = hidden_before.repeat(self.dec_layers, 1, 1)

        print('hidden_decoder:{}'.format(hidden_decoder.size()))

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

        print('Decoder Hidden')
        print(hidden.size())

        print('Encoder Outputs')
        print(encoder_outputs.size())

        print('----------------------------------3')

        # only pick up last layer hidden from decoder
        hidden = torch.unbind(hidden, dim=0)[0]
        # hidden = hidden[-1, :, :].squeeze(0)

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder output
        # encoder_outputs_left = encoder_outputs_left.permute(1, 0, 2)
        # print(encoder_outputs_left.size())
        # encoder_outputs_right = encoder_outputs_right.permute(1, 0, 2)
        # print(encoder_outputs_right.size())

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # print(hidden.size())
        # print(encoder_outputs.size())
        # print('-----------------')

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
        print('Decoder:')
        print('input:{}'.format(input.size()))

        input = input.unsqueeze(0)
        input = torch.unsqueeze(input, 2)

        embedded = self.dropout(torch.tanh(self.input_dec(input)))

        print('Decoder:')
        print('embedded:{}'.format(embedded.size()))

        # # only pick up last layer hidden
        # hidden = hidden[-1,:,:].squeeze(0)

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

        # print('input_dec:{}'.format(input_dec))
        # print('output:{}'.format(output))
        # print('weighted:{}'.format(weighted))

        output = self.out(torch.cat((output, weighted, input_dec), dim=1))

        # print('output:{}'.format(output))

        return output.squeeze(1), hidden, a


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

        outputs = torch.zeros(max_len, batch_size,
                              self.decoder.output_dim).to(self.device)

        # save attn states

        decoder_attn = torch.zeros(max_len, src_left.shape[0]).to(self.device)

        # print(outputs.size())

        # print(decoder_attn.size())

        # print('0')
        # print(src)

        # Shared Encoder
        encoder_outputs_left, encoder_outputs_right, hidden = self.shared_encoder(
            src_left, src_right)

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
            output, hidden, attn_weight = self.decoder(output, hidden,
                                                       encoder_outputs_left,
                                                       encoder_outputs_right)

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

        loss = train_iteration(model, optimizer, criterion, CLIP, WD,
                               x_train_batch, y_train_batch)

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
    iter_multiloss = [
        np.zeros(EPOCHS * iter_per_epoch),
        np.zeros(EPOCHS * iter_per_epoch),
        np.zeros(EPOCHS * iter_per_epoch)
    ]
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)
    perm_idx = np.random.permutation(X_test.shape[0])

    n_iter = 0

    with torch.no_grad():
        for t_i in range(0, X_test.shape[0], BATCH_SIZE):
            batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

            x_test_batch = np.take(X_test, batch_idx, axis=0)
            y_test_batch = np.take(y_test, batch_idx, axis=0)

            loss, mae, rmsle, rmse = evaluate_iteration(
                model, criterion, x_test_batch, y_test_batch)
            iter_losses[t_i // BATCH_SIZE] = loss
            iter_multiloss[0][t_i // BATCH_SIZE] = mae
            iter_multiloss[1][t_i // BATCH_SIZE] = rmsle
            iter_multiloss[2][t_i // BATCH_SIZE] = rmse

            # writer.add_scalars('Val_loss', {'val_loss': iter_losses[t_i // BATCH_SIZE]},
            #                    n_iter)

            n_iter += 1

    return np.mean(iter_losses[range(0, iter_per_epoch)]), np.mean(
        iter_multiloss[0][range(0, iter_per_epoch)]), np.mean(
            iter_multiloss[1][range(0, iter_per_epoch)]), np.mean(
                iter_multiloss[2][range(0, iter_per_epoch)])


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

    loss_mae = mean_absolute_error(y_test_numpy, output_numpy)
    loss_RMSLE = RMSLE(y_test_numpy, output_numpy)
    loss_RMSE = np.sqrt(mean_squared_error(y_test_numpy, output_numpy))

    # test_loss_meter.add(loss.item())

    # plot_result(output, y_test_tensor)
    # show_attention(x_test_tensor,output,decoder_attn)

    return loss.item(), loss_mae, loss_RMSLE, loss_RMSE
