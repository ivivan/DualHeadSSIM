import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import random, math, os, time

from utils.VLSW import pad_all_cases
# from VLSW import pad_all_cases
# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)


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


def preprocess_df(df):
    """ The training and testing data are manually selected.
    :param df:  dataframe with raw data
    :return:
    """

    df.set_index('datetime', inplace=True)

    ## some variables are not used in training the model, based on the performance evaluation
    df.drop(['chloro_con'], axis=1, inplace=True)

    tw = df['nitrate_con'].values.copy().reshape(-1, 1)

    # Standlization, use StandardScaler
    scaler_x = MinMaxScaler()
    scaler_x.fit(
        df[['temp_water', 'ph', 'spec_cond', 'diss_oxy_con', 'nitrate_con']])
    df[['temp_water', 'ph', 'spec_cond', 'diss_oxy_con',
        'nitrate_con']] = scaler_x.transform(df[[
            'temp_water', 'ph', 'spec_cond', 'diss_oxy_con', 'nitrate_con'
        ]])

    scaler_y = MinMaxScaler()
    scaler_y.fit(tw)
    y_all = scaler_y.transform(tw)

    # get data from 2014 and 2015
    # 6，7, 8, 9，10 as train; 11 as test

    df_train_one = df.loc['2014-06-01T00:00':'2014-10-31T23:30'].copy()
    df_train_two = df.loc['2015-06-01T00:00':'2015-10-31T23:30'].copy()

    df_test_one = df.loc['2014-11-01T00:00':'2014-11-30T23:30'].copy()
    df_test_two = df.loc['2015-11-01T00:00':'2015-11-30T23:30'].copy()

    return df_train_one, df_train_two, df_test_one, df_test_two, scaler_x, scaler_y

    # scaler_y = MinMaxScaler()
    # scaler_y.fit(tw)
    # y_all = scaler_y.transform(tw)


def train_val_test_generate(dataframe, model_params):
    '''
    :param dataframe: processed dataframe
    :param model_params: for input dim
    :return: train_x, train_y, test_x, test_y with the same length (by padding zero)
    '''

    train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples = pad_all_cases(
        dataframe, dataframe['nitrate_con'].values, model_params,
        model_params['min_before'], model_params['max_before'],
        model_params['min_after'], model_params['max_after'],
        model_params['output_length'])

    train_val_test_y = np.expand_dims(train_val_test_y, axis=2)

    return train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples


def train_test_split_SSIM(x, y, x_len, x_before_len, model_params, SEED):
    '''
    :param x: all x samples
    :param y: all y samples
    :param model_params: parameters
    :param SEED: random SEED
    :return: train set, test set
    '''

    ## check and remove samples with NaN (just incase)
    index_list = []
    for index, (x_s, y_s, len_s,
                len_before_s) in enumerate(zip(x, y, x_len, x_before_len)):
        if (np.isnan(x_s).any()) or (np.isnan(y_s).any()):
            index_list.append(index)

    x = np.delete(x, index_list, axis=0)
    y = np.delete(y, index_list, axis=0)
    x_len = np.delete(x_len, index_list, axis=0)
    x_before_len = np.delete(x_before_len, index_list, axis=0)

    print('x:{}'.format(x.shape))
    print('y:{}'.format(y.shape))

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=None,
                                                        random_state=SEED,
                                                        shuffle=False)

    x_train_len, x_test_len = train_test_split(x_len,
                                               test_size=None,
                                               random_state=SEED,
                                               shuffle=False)

    x_train_before_len, x_test_before_len = train_test_split(x_before_len,
                                                             test_size=None,
                                                             random_state=SEED,
                                                             shuffle=False)

    return x_train, y_train, x_train_len, x_train_before_len


def test_pm25_single_station():
    train_sampling_params = {
        'dim_in': 5,
        'output_length': 6,
        'min_before': 12,
        'max_before': 12,
        'min_after': 12,
        'max_after': 12,
        'file_path': '../data/simplified_PM25.csv'
    }

    test_sampling_params = {
        'dim_in': 5,
        'output_length': 6,
        'min_before': 12,
        'max_before': 12,
        'min_after': 12,
        'max_after': 12,
        'file_path': '../data/simplified_PM25.csv'
    }

    filepath = 'data/45_joined.csv'

    df = pd.read_csv(filepath, dayfirst=True)

    df_train_one, df_train_two, df_test_one, df_test_two, scaler_x, scaler_y = preprocess_df(
        df)

    # generate train/test samples seperately

    # train one
    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_train_one, train_sampling_params)

    x_train_one, y_train_one, x_train_len_one, x_train_before_len_one = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

    # train two
    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_train_two, train_sampling_params)

    x_train_two, y_train_two, x_train_len_two, x_train_before_len_two = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

    # concate all train data

    x_train = np.concatenate((x_train_one, x_train_two), axis=0)
    y_train = np.concatenate((y_train_one, y_train_two), axis=0)

    #------------------------------#

    # test one

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_test_one, test_sampling_params)

    x_test_one, y_test_one, x_test_len_one, x_test_before_len_one = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    # test two

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_test_two, test_sampling_params)

    x_test_two, y_test_two, x_test_len_two, x_test_before_len_two = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    # # concate all test data

    x_test = np.concatenate((x_test_one, x_test_two), axis=0)
    y_test = np.concatenate((y_test_one, y_test_two), axis=0)

    print('x_train:{}'.format(x_train.shape))
    print('y_train:{}'.format(y_train.shape))
    print('x_test:{}'.format(x_test.shape))
    print('y_test:{}'.format(y_test.shape))

    print('split train/test array')
    x_test_list = np.split(x_test, [12, 18], axis=1)
    x_train_list = np.split(x_train, [12, 18], axis=1)

    for i in x_test_list:
        print(i.shape)

    return (x_train, y_train), (x_test, y_test), (scaler_x, scaler_y)


if __name__ == "__main__":
    test_pm25_single_station()
