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


# def show_attention(input_sentence, output_words, attentions):
#     input_sentence = input_sentence.data.numpy()
#     output_words = output_words.data.numpy()

#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     # print('here')
#     # print(attentions.data.numpy())

#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)

#     # Set up axes
#     ax.set_xticklabels(input_sentence, rotation=90)
#     ax.set_yticklabels(output_words)

#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     # show_plot_visdom()


def show_attention(input_left, input_right, output_words, attentions):

    input_left = input_left.squeeze().data.numpy()
    input_right = input_right.squeeze().data.numpy()

    input_sentence = np.concatenate((input_left, input_right), axis=0)
    input_sentence = input_sentence[:, 3]

    output_words = output_words.data.numpy()

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.matshow(attentions.squeeze().numpy(), cmap='jet')
    fig.colorbar(cax)

    # set label with number
    x_tick = np.concatenate((np.arange(
        0, input_left.shape[0] + 1), np.arange(1, input_left.shape[0] + 1)),
                            axis=0)
    y_tick = np.arange(0, output_words.shape[0] + 1)

    ax.set_xticklabels(x_tick, rotation=90)
    ax.set_yticklabels(y_tick)

    # # Set up axes
    # ax.set_xticklabels(input_sentence, rotation=90)
    # ax.set_yticklabels(output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_aspect('auto')

    # show_plot_visdom()