import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from scipy import stats
from math import sqrt, fabs
import os
import argparse
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import itertools
import impyute
import datetime
from fancyimpute import BiScaler, NuclearNormMinimization, SoftImpute, MatrixFactorization, SimilarityWeightedAveraging
from fancyimpute import KNN
from tslearn.metrics import dtw, dtw_path

# fix random seed for reproducibility
seed = 1234
np.random.seed(seed)


def preprocessdf(df):

    df.set_index('Timestamp', inplace=True)

    # some variables are not used in training the model, based on the performance evaluation
    df.drop(['Dayofweek'], axis=1, inplace=True)
    df.drop(['Month'], axis=1, inplace=True)

    df = df.loc['2019-10-01T00:00':'2019-12-31T23:00'].copy()

    tw = df['NO3'].values.copy().reshape(-1, 1)

    # data clean, for short period of time
    # df.loc['2019-10-01T00:00':'2019-12-31T23:00', :] = df.loc['2019-10-01T00:00':'2019-12-31T23:00', :].fillna(method='ffill')

    scaler_x = MinMaxScaler()
    scaler_x.fit(
        df[['Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity', 'Level']])
    df[['Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity', 'Level']] = scaler_x.transform(df[[
        'Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity', 'Level'
    ]])

    scaler_y = MinMaxScaler()
    scaler_y.fit(tw)
    y_all = scaler_y.transform(tw)

    return df, scaler_x, scaler_y


def generate_test_samples(y, input_seq_len, output_seq_len):
    """
    Generate all the test samples at one time
    :param x: df
    :param y:
    :param input_seq_len:
    :param output_seq_len:
    :return:
    """
    total_samples = y.shape[0]

    input_batch_idxs = [list(range(i, i + input_seq_len+output_seq_len)) for i in
                        range((total_samples - input_seq_len - output_seq_len+1))]
    input_seq = np.take(y, input_batch_idxs, axis=0)

    return input_seq


# def rsquare(y_true,y_pred):
#     SS_res = K.sum(K.square(y_true - y_pred))
#     SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
#     return (1 - SS_res / (SS_tot + K.epsilon()))

def saveindex(index_list, path):
    with open(path, 'w') as file_handler:
        for item in index_list:
            file_handler.write("{}\n".format(item))


def readindex(path):
    temp = []
    with open(path) as index:
        for line in index:
            if line:
                temp.append(line)
    return temp


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    out = 0
    for i in range(y_true.shape[0]):
        a = y_true[i]
        b = y_pred[i]
        c = a+b
        if c == 0:
            continue
        out += fabs(a - b) / c
    out *= (200.0 / y_true.shape[0])
    return out


if __name__ == "__main__":

    # Default Parameters
    model_params = {
        'output_length': 3,
        'before': 10,
        'after': 10
    }

    filepath = 'data/QLD_nomiss.csv'
    df = pd.read_csv(filepath)
    df, scaler_x, scaler_y = preprocessdf(df)

    predictions = list()
    obs = list()
    model_inputs = list()

    loss_dtw = 0

    size = df.shape[0]-model_params['output_length'] - \
        model_params['before']-model_params['after']+1

    ##### match pytorch experiment #####

    for i in range(size):

        complete_matrix = df.iloc[i:i+model_params['output_length'] +
                                  model_params['before']+model_params['after'], :]
        complete_before = df.iloc[i:i+model_params['before'], :]
        complete_middle = df.iloc[i+model_params['before']:i +
                                  model_params['before']+model_params['output_length'], :]
        complete_after = df.iloc[i+model_params['before']+model_params['output_length']                                 :i+model_params['output_length']+model_params['before']+model_params['after'], :]

        incomplete_middle = complete_middle.copy()
        incomplete_middle[:] = np.nan

        # incomplete_middle[:] = np.nan

        # print(complete_before.shape)
        # print(complete_middle.shape)
        # print(complete_after.shape)

        target_column = complete_middle['NO3'].values.copy()

        incomplete_matrix = np.concatenate(
            (complete_before, incomplete_middle, complete_after), axis=0)

        complete_matrix = np.concatenate(
            (complete_before, complete_middle, complete_after), axis=0)

        # ########### get original input x ##########
        ori_inputs = scaler_x.inverse_transform(complete_matrix)

        # NO3 on 3rd column, water level on 6th column in csv
        model_inputs.append(ori_inputs[:, 2])

        # ##impulation
        # ## KNN
        filled = impyute.imputation.cs.fast_knn(
            incomplete_matrix, k=10)  # impyute implement

        # ## EM
        # filled = impyute.imputation.cs.em(incomplete_matrix,loops=1000)

        # ## MICE
        # filled = impyute.imputation.cs.mice(incomplete_matrix)

        # ## Mean
        # filled = impyute.imputation.cs.mean(incomplete_matrix)

        # ## LOCF
        # filled = impyute.imputation.ts.locf(incomplete_matrix,axis=1)

        # ## Linear
        # incomplete_dataframe = pd.DataFrame(incomplete_matrix)
        # incomplete_dataframe = incomplete_dataframe.interpolate(method='linear')
        # filled = incomplete_dataframe.to_numpy()

        pred = filled[model_params['before']:model_params['before']+model_params['output_length'], 2]

        loss_dtw += dtw(target_column, pred)

        obs.append(target_column.tolist())
        predictions.append(pred.tolist())

    # ######### plot results ############
    # for z in range(len(model_inputs)):

    #     # print(z)

    #     x = np.arange(len(model_inputs[z]))
    #     ori_in = model_inputs[z]

    #     pred_out = ori_in.copy()
    #     arr = np.array(predictions[z])
    #     y_pred = scaler_y.inverse_transform(arr.reshape(1,-1))

    #     pred_out[10:13] = y_pred

    #     plt.figure()
    #     # plt.plot(pred_list, label='Predicted')
    #     # plt.plot(ori_list, label="True")
    #     plt.scatter(x, pred_out, label='Predicted')
    #     plt.scatter(x, ori_in, label="True")
    #     plt.legend(loc='upper left')
    #     plt.show()

    predictions = [item for sublist in predictions for item in sublist]
    obs = [item for sublist in obs for item in sublist]

    # ######### save imputation values #############

    # predictions_saved = np.asarray(predictions)
    # np.save('./results/{}_outputs_scal'.format('knn_NO3_1012'), predictions_saved)

    print(len(predictions))
    print(len(obs))

    print("-------------")
    print("RMSE:")
    mse = mean_squared_error(obs, predictions)
    rmse = sqrt(mse)
    print(rmse)

    print("--------")
    print("MAE:")
    mae = mean_absolute_error(obs, predictions)
    print(mae)
    print("---------")
    # print("R2:")
    # r2 = r2_score(obs, predictions)
    # print(r2)
    # print("---------")
    # mape = mean_absolute_percentage_error(obs, predictions)
    # print("MAPE (sklearn):{0:f}".format(mape))
    # print("---------")
    # smape = symmetric_mean_absolute_percentage_error(np.array(obs), np.array(predictions))
    # print("SMAPE (sklearn):{0}".format(smape))
    print("DTW:")
    dtw = loss_dtw/size
    print(dtw)