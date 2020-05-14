import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from math import sqrt,fabs
import os
# import h5py
import argparse
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf,pacf
import itertools
import impyute
import datetime
from fancyimpute import BiScaler, NuclearNormMinimization, SoftImpute,MatrixFactorization,SimilarityWeightedAveraging
from fancyimpute import KNN
from tslearn.metrics import dtw, dtw_path

# fix random seed for reproducibility
seed = 1234
np.random.seed(seed)


def preprocessdf(df):



    df.set_index('Timestamp', inplace=True)

    ## some variables are not used in training the model, based on the performance evaluation
    df.drop(['Dayofweek'], axis=1, inplace=True)
    df.drop(['Month'], axis=1, inplace=True)



    df = df.loc['2019-10-01T00:00':'2019-12-31T23:00'].copy()

    ## data clean, for short period of time
    df.loc['2019-10-01T00:00':'2019-12-31T23:00', :] = df.loc['2019-10-01T00:00':'2019-12-31T23:00', :].fillna(method='ffill')



    scaler_x = MinMaxScaler()
    scaler_x.fit(
        df[['Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity','Level']])
    df[['Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity','Level']] = scaler_x.transform(df[[
            'Q', 'Conductivity', 'NO3', 'Temp', 'Turbidity','Level'
        ]])



    # target = df['NO3'].values.copy()
        
    return df



    # temp_all = df.copy()

    # incomplete = df.copy()

    # startdate = datetime.date(year, month, day) + datetime.timedelta(loop)
    # period = datetime.timedelta(length)

    # enddate = startdate + period

    # incomplete.loc[startdate:enddate] = np.nan
    # sub_incomplete = incomplete.loc[startdate-datetime.timedelta(3):enddate+datetime.timedelta(1)]

    # each_N = temp_all.loc[startdate:enddate]['N'].values.copy()

    # print(each_N)


    # return temp_all,incomplete,sub_incomplete, each_N


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


def rsquare(y_true,y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

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
        'output_length':6,
        'before':10,
        'after':10
    }

    filepath = 'data/QLD_nomiss.csv'
    df = pd.read_csv(filepath)
    df = preprocessdf(df)

    predictions = list()
    obs = list()

    loss_dtw = 0

    size = df.shape[0]-model_params['output_length']-model_params['before']-model_params['after']+1

    for i in range(size):
    #     print(i)
    # for i in range(12):

        complete_matrix = df.iloc[i:i+model_params['output_length']+model_params['before']+model_params['after'],:]

        complete_before = df.iloc[i:i+model_params['before'],:]
        complete_middle = df.iloc[i+model_params['before']:i+model_params['before']+model_params['output_length'],:]
        complete_after = df.iloc[i+model_params['before']+model_params['output_length']:i+model_params['output_length']+model_params['before']+model_params['after'],:]

        incomplete_middle = complete_middle.copy()
        incomplete_middle[:] = np.nan

        # print(complete_before.shape)
        # print(complete_middle.shape)
        # print(complete_after.shape)



        target_column = complete_middle['NO3'].values.copy()


        incomplete_matrix = np.concatenate((complete_before, incomplete_middle,complete_after), axis=0)

        complete_matrix = np.concatenate((complete_before, complete_middle,complete_after), axis=0)


        # ##impulation
        # ## KNN
        # filled = KNN(k=3).fit_transform(incomplete_matrix)
        # filled  = impyute.fast_knn(incomplete_matrix)

        # ## EM
        # filled = impyute.em(incomplete_matrix)


        # ## MICE
        # filled = impyute.imputation.cs.mice(incomplete_matrix)


        # ## Mean
        # filled = impyute.imputation.cs.mean(incomplete_matrix)

        # ## LOCF
        # filled = impyute.imputation.ts.locf(incomplete_matrix,axis=1)

        ## Linear
        incomplete_dataframe = pd.DataFrame(incomplete_matrix)
        incomplete_dataframe = incomplete_dataframe.interpolate(method='linear')
        filled = incomplete_dataframe.to_numpy()


        # knnImpute = KNN(k=3)
        # filled = knnImpute.complete(input_matrix)

        pred = filled[model_params['before']:model_params['before']+model_params['output_length'],2]

        loss_dtw += dtw(target_column,pred)

        # print('pred:{}'.format(len(pred)))
        # print('obs:{}'.format(len(target_column)))

        obs.append(target_column.tolist())
        predictions.append(pred.tolist())


    predictions = [item for sublist in predictions for item in sublist]
    obs = [item for sublist in obs for item in sublist]

    # print(len(predictions))
    # print(len(obs))

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



