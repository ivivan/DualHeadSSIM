from tslearn.metrics import dtw, dtw_path
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from warmup_scheduler import GradualWarmupScheduler
from utils.cyclic_scheduler import CyclicLRWithRestarts
from utils.dilate_loss import dilate_loss
from utils.adamw import AdamW
from utils.metrics import RMSLE
from utils.support import *
from utils.prepare_QLD import test_qld_single_station
from utils.early_stopping import EarlyStopping
from models.DualHead_NoShare import Shared_Encoder, Cross_Attention, Decoder, DualSSIM
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
import os
import time
from torch.optim.lr_scheduler import StepLR, ExponentialLR

import numpy as np
np.set_printoptions(threshold=np.inf)


# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, criterion, X_train_left, X_train_right, y_train):

    iter_per_epoch = int(np.ceil(X_train_left.shape[0] * 1. / BATCH_SIZE))
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)

    n_iter = 0

    perm_idx = np.random.permutation(X_train_left.shape[0])

    # train for each batch

    for t_i in range(0, X_train_left.shape[0], BATCH_SIZE):
        batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

        x_train_left_batch = np.take(X_train_left, batch_idx, axis=0)
        x_train_right_batch = np.take(X_train_right, batch_idx, axis=0)
        y_train_batch = np.take(y_train, batch_idx, axis=0)

        loss = train_iteration(model, optimizer, criterion, CLIP,
                               x_train_left_batch, x_train_right_batch,
                               y_train_batch)

        iter_losses[t_i // BATCH_SIZE] = loss

        n_iter += 1

    return np.mean(iter_losses[range(0, iter_per_epoch)])


def train_iteration(model, optimizer, criterion, clip, X_train_left,
                    X_train_right, y_train):
    model.train()
    optimizer.zero_grad()

    X_train_left = np.transpose(X_train_left, [1, 0, 2])
    X_train_right = np.transpose(X_train_right, [1, 0, 2])
    y_train = np.transpose(y_train, [1, 0, 2])

    X_train_left_tensor = numpy_to_tvar(X_train_left)
    X_train_right_tensor = numpy_to_tvar(X_train_right)
    y_train_tensor = numpy_to_tvar(y_train)

    output, atten = model(X_train_left_tensor,
                          X_train_right_tensor, y_train_tensor)

    output = output.permute(1, 0, 2)
    y_train_tensor = y_train_tensor.permute(1, 0, 2)

    loss_mse, loss_shape, loss_temporal = torch.tensor(
        0), torch.tensor(0), torch.tensor(0)

    loss, loss_shape, loss_temporal = dilate_loss(
        y_train_tensor, output, 0.85, 0.01, device)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    # # for AdamW+Cyclical Learning Rate
    # scheduler.batch_step()

    # loss_meter.add(loss.item())

    return loss.item()


# evaluate


def evaluate(model, criterion, X_test_left, X_test_right, y_test):

    epoch_loss = 0
    iter_per_epoch = int(np.ceil(X_test_left.shape[0] * 1. / BATCH_SIZE))
    iter_losses = np.zeros(EPOCHS * iter_per_epoch)
    iter_multiloss = [np.zeros(EPOCHS * iter_per_epoch), np.zeros(EPOCHS * iter_per_epoch),
                      np.zeros(EPOCHS * iter_per_epoch), np.zeros(EPOCHS * iter_per_epoch)]
    perm_idx = np.random.permutation(X_test_left.shape[0])

    n_iter = 0

    with torch.no_grad():
        for t_i in range(0, X_test_left.shape[0], BATCH_SIZE):
            batch_idx = perm_idx[t_i:(t_i + BATCH_SIZE)]

            x_test_left_batch = np.take(X_test_left, batch_idx, axis=0)
            x_test_right_batch = np.take(X_test_right, batch_idx, axis=0)
            y_test_batch = np.take(y_test, batch_idx, axis=0)

            loss, mae, rmsle, rmse, loss_tdi = evaluate_iteration(model, criterion, x_test_left_batch,
                                                                  x_test_right_batch, y_test_batch)
            iter_losses[t_i // BATCH_SIZE] = loss
            iter_multiloss[0][t_i // BATCH_SIZE] = mae
            iter_multiloss[1][t_i // BATCH_SIZE] = rmsle
            iter_multiloss[2][t_i // BATCH_SIZE] = rmse
            iter_multiloss[3][t_i // BATCH_SIZE] = loss_tdi

            n_iter += 1

    return np.mean(iter_losses[range(0, iter_per_epoch)]), np.mean(iter_multiloss[0][range(0, iter_per_epoch)]), np.mean(
        iter_multiloss[1][range(0, iter_per_epoch)]), np.mean(iter_multiloss[2][range(0, iter_per_epoch)]), np.mean(iter_multiloss[3][range(0, iter_per_epoch)])


def evaluate_iteration(model, criterion, X_test_left, X_test_right, y_test):
    model.eval()

    x_test_left = np.transpose(X_test_left, [1, 0, 2])
    x_test_right = np.transpose(X_test_right, [1, 0, 2])
    y_test = np.transpose(y_test, [1, 0, 2])

    x_test_left_tensor = numpy_to_tvar(x_test_left)
    x_test_right_tensor = numpy_to_tvar(x_test_right)

    y_test_tensor = numpy_to_tvar(y_test)

    output, atten = model(x_test_left_tensor,
                          x_test_right_tensor, y_test_tensor, 0)

    loss = criterion(output, y_test_tensor)
    loss_mse, loss_dtw, loss_tdi = 0, 0, 0
    loss_mae, loss_RMSLE, loss_RMSE = 0, 0, 0

    for k in range(BATCH_SIZE):
        target_k_cpu = y_test_tensor[:, k, 0:1].view(-1).detach().cpu().numpy()
        output_k_cpu = output[:, k, 0:1].view(-1).detach().cpu().numpy()

        loss_dtw += dtw(target_k_cpu, output_k_cpu)
        path, sim = dtw_path(target_k_cpu, output_k_cpu)

        Dist = 0
        for i, j in path:
            Dist += (i-j)*(i-j)
        loss_tdi += Dist / (N_output*N_output)

        loss_mae += mean_absolute_error(target_k_cpu, output_k_cpu)
        loss_RMSLE += np.sqrt(mean_squared_error(target_k_cpu, output_k_cpu))
        loss_RMSE += np.sqrt(mean_squared_error(target_k_cpu, output_k_cpu))

    loss_dtw = loss_dtw / BATCH_SIZE
    loss_tdi = loss_tdi / BATCH_SIZE
    loss_mae = loss_mae / BATCH_SIZE
    loss_RMSLE = loss_RMSLE / BATCH_SIZE
    loss_RMSE = loss_RMSE / BATCH_SIZE

    # # metric
    # output_numpy = output.cpu().data.numpy()
    # y_test_numpy = y_test_tensor.cpu().data.numpy()

    # loss_mae = mean_absolute_error(y_test_numpy,output_numpy)
    # loss_RMSLE = np.sqrt(mean_squared_error(y_test_numpy,output_numpy))
    # loss_RMSE = np.sqrt(mean_squared_error(y_test_numpy,output_numpy))

    # test_loss_meter.add(loss.item())

    # plot_result(output, y_test_tensor)
    # show_attention(x_test_left_tensor, x_test_right_tensor,output,atten)
    # plt.show()

    return loss.item(), loss_mae, loss_RMSLE, loss_RMSE, loss_dtw


def predict_ts(model, X_test_left, X_test_right, scaler_y, max_gap_size=6, BATCH_SIZE=1, device=device):
    model.eval()

    with torch.no_grad():

        x_test_left = np.transpose(X_test_left, [1, 0, 2])
        x_test_right = np.transpose(X_test_right, [1, 0, 2])

        empty_y_tensor = torch.zeros(max_gap_size, BATCH_SIZE,
                                     1).to(device)

        x_test_left_tensor = numpy_to_tvar(x_test_left)
        x_test_right_tensor = numpy_to_tvar(x_test_right)

        output, _ = model(x_test_left_tensor,
                          x_test_right_tensor, empty_y_tensor, 0)

        output = torch.squeeze(output)
        output = torch.transpose(output, 0, 1)
        output = torch.flatten(output)

        # scalar
        output_numpy = output.cpu().data.numpy()
        output_numpy_origin = scaler_y.inverse_transform(
            output_numpy.reshape(-1, 1))

    return output_numpy_origin, output_numpy


if __name__ == "__main__":

    # model hyperparameters
    INPUT_DIM = 6
    OUTPUT_DIM = 1
    ENC_HID_DIM = 50
    DEC_HID_DIM = 50
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    ECN_Layers = 1
    DEC_Layers = 1
    LR = 0.001  # learning rate
    CLIP = 1
    EPOCHS = 500
    BATCH_SIZE = 1
    N_output = 6

    # Different test data

    (x_train, y_train), (x_test, y_test), (scaler_x,
                                           scaler_y) = test_qld_single_station()

    print('split train/test array')
    x_test_list = np.split(x_test, [10, 16], axis=1)
    x_train_list = np.split(x_train, [10, 16], axis=1)

    # Split input into two

    X_train_left = x_train_list[0]
    X_train_right = x_train_list[2]
    X_test_left = x_test_list[0]
    X_test_right = x_test_list[2]

    print('X_train_left:{}'.format(X_train_left.shape))
    print('X_train_right:{}'.format(X_train_right.shape))
    print('X_test_left:{}'.format(X_test_left.shape))
    print('X_test_right:{}'.format(X_test_right.shape))

    # fit for batchsize  check dataloader droplast
    X_train_left = X_train_left[:]
    X_train_right = X_train_right[:]
    # X_test_left = X_test_left[:2180]
    # X_test_right = X_test_right[:2180]

    # Model
    cross_attn = Cross_Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Shared_Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ECN_Layers,
                         DEC_Layers, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_Layers,
                  DEC_DROPOUT, cross_attn)

    model = DualSSIM(enc, dec, device).to(device)
    model.apply(init_weights)

    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    # Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                             step_size=10,
    #                                             gamma=0.1)

    # optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # scheduler = CyclicLRWithRestarts(optimizer, BATCH_SIZE, 3202, restart_period=5, t_mult=1.2, policy="cosine")

    # warmup
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    scheduler_steplr = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=5)

    criterion = nn.MSELoss()

    # # visulization visdom
    # vis = Visualizer(env='attention')
    # loss_meter = meter.AverageValueMeter()
    # test_loss_meter = meter.AverageValueMeter()

    # Early Stopping
    # initialize the early_stopping object
    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 10
    early_stopping = EarlyStopping(output_path='checkpoints/Nitrate6_0103.pt',
                                   patience=patience,
                                   verbose=True)

    optimizer.zero_grad()
    optimizer.step()

    ###################### training ###################
    # best_valid_loss = float('inf')
    # for epoch in range(EPOCHS):

    #     scheduler_warmup.step(epoch)
    #     train_epoch_losses = np.zeros(EPOCHS)
    #     evaluate_epoch_losses = np.zeros(EPOCHS)
    #     # loss_meter.reset()

    #     # print('Epoch:', epoch, 'LR:', scheduler.get_lr())

    #     start_time = time.time()
    #     train_loss = train(model, optimizer, criterion, X_train_left,
    #                        X_train_right, y_train)
    #     valid_loss,test_mae, test_rmsle, test_rmse, test_tdi = evaluate(model, criterion, X_test_left, X_test_right,
    #                           y_test)
    #     end_time = time.time()

    #     # # visulization
    #     # vis.plot_many_stack({'train_loss': loss_meter.value()[0], 'test_loss': test_loss_meter.value()[0]})

    #     train_epoch_losses[epoch] = train_loss
    #     evaluate_epoch_losses[epoch] = valid_loss

    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    #     # early_stopping needs the validation loss to check if it has decresed,
    #     # and if it has, it will make a checkpoint of the current model
    #     early_stopping(valid_loss, model)

    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         break

    #     print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    #     print(
    #         f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}'
    #     )
    #     print(
    #         f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}'
    #     )
    #     print(f'| MAE: {test_mae:.4f} | Test PPL: {math.exp(test_mae):7.4f} |')
    #     print(f'| RMSLE: {test_rmsle:.4f} | Test PPL: {math.exp(test_rmsle):7.4f} |')
    #     print(f'| RMSE: {test_rmse:.4f} | Test PPL: {math.exp(test_rmse):7.4f} |')
    #     print(f'| TDI: {test_tdi:.4f} | Test PPL: {math.exp(test_tdi):7.4f} |')

    # # # prediction

    # # get one sample for attention visulization
    # X_test_left = X_test_left[5:6,:,:]
    # X_test_right = X_test_right[5:6,:,:]
    # y_test = y_test[5:6,:,:]

    ####### load and evaluate ####################

    model.load_state_dict(torch.load('checkpoints/Nitrate6_1012.pt'))

    test_loss, test_mae, test_rmsle, test_rmse, test_tdi = evaluate(
        model, criterion, X_test_left, X_test_right, y_test)

    print(
        f'| Test Loss: {test_loss:.4f} | Test PPL: {math.exp(test_loss):7.4f} |')
    print(f'| MAE: {test_mae:.4f} | Test PPL: {math.exp(test_mae):7.4f} |')
    print(
        f'| RMSLE: {test_rmsle:.4f} | Test PPL: {math.exp(test_rmsle):7.4f} |')
    print(f'| RMSE: {test_rmse:.4f} | Test PPL: {math.exp(test_rmse):7.4f} |')
    print(f'| DTW: {test_tdi:.4f} | Test PPL: {math.exp(test_tdi):7.4f} |')

    ######## check imputation value #######

    outputs_ori, outputs_scal = predict_ts(
        model, X_test_left, X_test_right, scaler_y, max_gap_size=6, BATCH_SIZE=1, device=device)

    print('*************')
    print('outputs_ori:{}'.format(outputs_ori.shape))
    print('*************')
    print('outputs_scal:{}'.format(outputs_scal.shape))

    np.save('./results/{}_ori'.format('Nitrate6_1012'), outputs_ori)
    np.save('./results/{}_scal'.format('Nitrate6_1012'), outputs_scal)

    # ######### loop plot test data ###############

    # # for i in range(0,X_test_left.shape[0]):
    # for i in range(0,1):

    #     X_test_left_pick = X_test_left[i,:,:]
    #     X_test_right_pick = X_test_right[i,:,:]
    #     y_test_pick = y_test[i,:,:]

    #     X_test_left_pick = np.expand_dims(X_test_left_pick, axis=0)
    #     X_test_right_pick = np.expand_dims(X_test_right_pick, axis=0)
    #     y_test_pick = np.expand_dims(y_test_pick, axis=0)

    #     print(X_test_left_pick.shape)
    #     print(X_test_right_pick.shape)
    #     print(y_test_pick.shape)

    #     outputs_ori, outputs_scal = predict_ts(model, X_test_left_pick, X_test_right_pick, scaler_y, max_gap_size=3, BATCH_SIZE=1,device=device)
    #     print('*************')
    #     X_test_left_pick = scaler_x.inverse_transform(X_test_left_pick[0])
    #     X_test_right_pick = scaler_x.inverse_transform(X_test_right_pick[0])
    #     y_test_pick = scaler_y.inverse_transform(y_test_pick.reshape(1,-1))

    #     print(X_test_left_pick[:,2])
    #     print(X_test_right_pick[:,2])
    #     print(y_test_pick[0])

    #     print('*************')
    #     print('outputs_ori:{}'.format(outputs_ori))
    #     print('*************')
    #     print('outputs_scal:{}'.format(outputs_scal))

    #     outputs_ori = [item for sublist in outputs_ori for item in sublist]

    #     list_before = X_test_left_pick[:,2].tolist()
    #     list_middle = y_test_pick[0].tolist()
    #     list_after = X_test_right_pick[:,2].tolist()

    #     print('outputs_ori:{}'.format(outputs_ori))

    #     ori_list = list_before + list_middle + list_after
    #     pred_list = list_before + outputs_ori + list_after

    #     x = np.arange(len(ori_list))

    #     plt.figure()
    #     # plt.plot(pred_list, label='Predicted')
    #     # plt.plot(ori_list, label="True")
    #     plt.scatter(x, pred_list, label='Predicted')
    #     plt.scatter(x, ori_list, label="True")
    #     plt.legend(loc='upper left')
    #     plt.show()
    #     # plt.pause(0.0001)
