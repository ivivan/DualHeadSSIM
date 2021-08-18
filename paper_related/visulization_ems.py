from fancyimpute import BiScaler, NuclearNormMinimization, SoftImpute, MatrixFactorization, SimilarityWeightedAveraging
from statsmodels.tsa.stattools import acf, pacf
from utils.prepare_QLD import test_qld_single_station
from tslearn.metrics import dtw, dtw_path
from fancyimpute import KNN
import datetime
import impyute
import itertools
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from math import sqrt, fabs
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)

# import h5py


# fix random seed for reproducibility
seed = 1234
np.random.seed(seed)


if __name__ == "__main__":

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

    # Model_list = ['Dual_SSIM', 'SSIM', 'brits_i', 'm_rnn', 'em', 'knn']
    Model_list = ['Dual_SSIM', 'SSIM', 'brits_i', 'm_rnn', 'em', 'knn', 'Linear', 'mice', 'LOCF', 'mean']

    level_imputed_list = []
    NO3_imputed_list = []

    ########### read all imputations for 10 models #############
    for l in Model_list:
        level_imputed_scal = f'results/EMS/{l}_Nitrate6_1012_outputs_scal.npy'
        NO3_imputed_scal = f'results/EMS/{l}_Nitrate6_1012_outputs_scal.npy'

        level_imputed_array = np.load(level_imputed_scal)
        NO3_imputed_array = np.load(NO3_imputed_scal)

        level_imputed_array_reshape = np.array_split(
            level_imputed_array, level_imputed_array.shape[0]//6)   ### 6 output size
        NO3_imputed_array_reshape = np.array_split(
            NO3_imputed_array, NO3_imputed_array.shape[0]//6)

        level_imputed_list.append(level_imputed_array_reshape)
        NO3_imputed_list.append(NO3_imputed_array_reshape)

    print(len(level_imputed_list[3]))

    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    ######### loop plot test data ###############

    for i in range(0, X_test_left.shape[0]):
        # for i in range(0,1):

        X_test_left_pick = X_test_left[i, :, :]
        X_test_right_pick = X_test_right[i, :, :]
        y_test_pick = y_test[i, :, :]

        X_test_left_pick = np.expand_dims(X_test_left_pick, axis=0)
        X_test_right_pick = np.expand_dims(X_test_right_pick, axis=0)
        y_test_pick = np.expand_dims(y_test_pick, axis=0)

        print(X_test_left_pick.shape)
        print(X_test_right_pick.shape)
        print(y_test_pick.shape)

        print('*************')
        X_test_left_pick = scaler_x.inverse_transform(X_test_left_pick[0])
        X_test_right_pick = scaler_x.inverse_transform(X_test_right_pick[0])
        y_test_pick = scaler_y.inverse_transform(y_test_pick.reshape(1, -1))

        ######### imputed value #######

        imputed_Dual_SSIM = scaler_y.inverse_transform(
            level_imputed_list[0][i].reshape(-1, 1))
        imputed_SSIM = scaler_y.inverse_transform(
            level_imputed_list[1][i].reshape(-1, 1))
        imputed_BRITS = scaler_y.inverse_transform(
            level_imputed_list[2][i].reshape(-1, 1))
        imputed_MRNN = scaler_y.inverse_transform(
            level_imputed_list[3][i].reshape(-1, 1))
        imputed_EM = scaler_y.inverse_transform(
            level_imputed_list[4][i].reshape(-1, 1))
        imputed_KNN = scaler_y.inverse_transform(
            level_imputed_list[5][i].reshape(-1, 1))
        imputed_Linear = scaler_y.inverse_transform(
            level_imputed_list[6][i].reshape(-1, 1))
        imputed_MICE = scaler_y.inverse_transform(
            level_imputed_list[7][i].reshape(-1, 1))
        imputed_LOCF = scaler_y.inverse_transform(
            level_imputed_list[8][i].reshape(-1, 1))
        imputed_MEAN = scaler_y.inverse_transform(
            level_imputed_list[9][i].reshape(-1, 1))

        imputed_Dual_SSIM_ori = [
            item for sublist in imputed_Dual_SSIM for item in sublist]
        imputed_SSIM_ori = [
            item for sublist in imputed_SSIM for item in sublist]
        imputed_BRITS_ori = [
            item for sublist in imputed_BRITS for item in sublist]
        imputed_MRNN_ori = [
            item for sublist in imputed_MRNN for item in sublist]
        imputed_EM_ori = [item for sublist in imputed_EM for item in sublist]
        imputed_KNN_ori = [item for sublist in imputed_KNN for item in sublist]
        imputed_Linear_ori = [item for sublist in imputed_Linear for item in sublist]
        imputed_MICE_ori = [item for sublist in imputed_MICE for item in sublist]
        imputed_LOCF_ori = [item for sublist in imputed_LOCF for item in sublist]
        imputed_MEAN_ori = [item for sublist in imputed_MEAN for item in sublist]

        print(X_test_left_pick[:, 2])
        print(X_test_right_pick[:, 2])
        print(y_test_pick[0])

        # dim 5 --level    dim 2---NO3
        list_before = X_test_left_pick[:, 2].tolist()
        list_middle = y_test_pick[0].tolist()
        list_after = X_test_right_pick[:, 2].tolist()

        ori_list = list_before + list_middle + list_after

        ###### 6 predictions ########
        pred_list_Dual_SSIM = list_before + imputed_Dual_SSIM_ori + list_after
        pred_list_SSIM = list_before + imputed_SSIM_ori + list_after
        pred_list_BRITS = list_before + imputed_BRITS_ori + list_after
        pred_list_MRNN = list_before + imputed_MRNN_ori + list_after
        pred_list_EM = list_before + imputed_EM_ori + list_after
        pred_list_KNN = list_before + imputed_KNN_ori + list_after
        pred_list_Linear = list_before + imputed_Linear_ori + list_after
        pred_list_MICE = list_before + imputed_MICE_ori + list_after
        pred_list_LOCF = list_before + imputed_LOCF_ori + list_after
        pred_list_MEAN = list_before + imputed_MEAN_ori + list_after

        x = np.arange(len(ori_list))

        fig, ax = plt.subplots()
        ax.plot(pred_list_Dual_SSIM,
                color=tableau20[0], label='Predicted (Dual-SSIM)')
        ax.plot(pred_list_SSIM, color=tableau20[1], label='Predicted (SSIM)')
        ax.plot(pred_list_BRITS, color=tableau20[2], label='Predicted (BRITS)')
        ax.plot(pred_list_MRNN, color=tableau20[3], label='Predicted (M-RNN)')
        ax.plot(pred_list_EM, color=tableau20[4], label='Predicted (EM)')
        ax.plot(pred_list_KNN, color=tableau20[5], label='Predicted (KNN)')
        ax.plot(pred_list_Linear, color=tableau20[6], label='Predicted (Linear)')
        ax.plot(pred_list_MICE, color=tableau20[7], label='Predicted (MICE)')
        ax.plot(pred_list_LOCF, color=tableau20[8], label='Predicted (LOCF)')
        ax.plot(pred_list_MEAN, color=tableau20[9], label='Predicted (Mean)')
        ax.plot(ori_list, color=tableau20[10], label="Ground Truth")

        ax.legend(loc='upper right', fontsize=18)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Nitrate Concentration (mg/L)', fontsize=18)

        plt.show()
