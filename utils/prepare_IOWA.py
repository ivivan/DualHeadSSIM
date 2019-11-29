import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random, math, os, time

from VLSW import pad_all_cases

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

    tw = df['diss_oxy_con'].values.copy().reshape(-1, 1)

    # Standlization, use StandardScaler
    scaler_x = StandardScaler()
    scaler_x.fit(
        df[['temp_water', 'ph', 'spec_cond', 'diss_oxy_con', 'nitrate_con']])
    df[['temp_water', 'ph', 'spec_cond', 'diss_oxy_con',
        'nitrate_con']] = scaler_x.transform(df[[
            'temp_water', 'ph', 'spec_cond', 'diss_oxy_con', 'nitrate_con'
        ]])

    scaler_y = StandardScaler()
    scaler_y.fit(tw)
    y_all = scaler_y.transform(tw)

    df_2016 = df.loc['2016-03-10T00:00':'2016-11-13T23:30'].copy()
    df_2017 = df.loc['2017-02-14T00:00':'2017-12-04T23:30'].copy()

    print(df_2016.head())
    print(df_2017.head())

    return df_2016, df_2017, y_all, scaler_x, scaler_y


def train_val_test_generate(dataframe, model_params):
    '''
    :param dataframe: processed dataframe
    :param model_params: for input dim
    :return: train_x, train_y, test_x, test_y with the same length (by padding zero)
    '''

    # generate samples
    y = dataframe['diss_oxy_con'].copy()
    y = np.expand_dims(y, axis=2)

    train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples = \
        series_to_superviesed(dataframe.to_numpy(), y, model_params['min_before'], model_params[
            'output_length'], split=None)

    # print('train_val_test_x:{}'.format(train_val_test_x.shape))

    len_x_samples = np.full((train_val_test_x.shape[0], 1),
                            train_val_test_x.shape[1])

    # print('len_x_samples:{}'.format(len_x_samples.shape))
    # print(len_x_samples)

    return train_val_test_x, train_val_test_y, len_x_samples, len_x_samples


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
                                                        train_size=0.9,
                                                        random_state=SEED,
                                                        shuffle=False)

    x_train_len, x_test_len = train_test_split(x_len,
                                               train_size=0.9,
                                               random_state=SEED,
                                               shuffle=False)

    x_train_before_len, x_test_before_len = train_test_split(x_before_len,
                                                             train_size=0.9,
                                                             random_state=SEED,
                                                             shuffle=False)

    return x_train, x_test, y_train, y_test, x_train_len, x_train_before_len


def test_pm25_single_station():
    train_sampling_params = {
        'dim_in': 5,
        'output_length': 48,
        'min_before': 96,
        'max_before': 96,
        'min_after': 0,
        'max_after': 0,
        'file_path': '../data/simplified_PM25.csv'
    }

    test_sampling_params = {
        'dim_in': 11,
        'output_length': 6,
        'min_before': 48,
        'max_before': 48,
        'min_after': 0,
        'max_after': 0,
        'file_path': '../data/simplified_PM25.csv'
    }

    filepath = '../data/iowa/Bias_attention.csv'

    df = pd.read_csv(filepath)

    df_2016, df_2017, y, scaler_x, scaler_y = preprocess_df(df)

    # sample 2016
    x_samples_2016, y_samples_2016, x_len_2016, x_before_len_2016 = train_val_test_generate(
        df_2016, train_sampling_params)

    print('X_samples:{}'.format(x_samples_2016.shape))
    print('y_samples:{}'.format(y_samples_2016.shape))

    x_train_2016, x_test_2016, y_train_2016, y_test_2016, x_train_len_2016, x_train_before_len_2016 = train_test_split_SSIM(
        x_samples_2016, y_samples_2016, x_len_2016, x_before_len_2016,
        train_sampling_params, SEED)

    print('x_train:{}'.format(x_train_2016.shape))
    print('y_train:{}'.format(y_train_2016.shape))
    print('x_train_len:{}'.format(x_train_len_2016.shape))
    print('x_train_before_len:{}'.format(x_train_before_len_2016.shape))

    x_train_2016 = x_train_2016[:26200, :, :]
    y_train_2016 = y_train_2016[:26200, :, :]

    x_train_len_2016 = x_train_len_2016[:26200]
    x_train_before_len_2016 = x_train_before_len_2016[:26200]

    # sample 2017
    x_samples_2017, y_samples_2017, x_len_2017, x_before_len_2017 = train_val_test_generate(
        df_2017, train_sampling_params)

    print('X_samples:{}'.format(x_samples_2017.shape))
    print('y_samples:{}'.format(y_samples_2017.shape))

    x_train_2017, x_test_2017, y_train_2017, y_test_2017, x_train_len_2017, x_train_before_len_2017 = train_test_split_SSIM(
        x_samples_2017, y_samples_2017, x_len_2017, x_before_len_2017,
        train_sampling_params, SEED)

    print('x_train:{}'.format(x_train_2017.shape))
    print('y_train:{}'.format(y_train_2017.shape))
    print('x_train_len:{}'.format(x_train_len_2017.shape))
    print('x_train_before_len:{}'.format(x_train_before_len_2017.shape))

    x_train_2017 = x_train_2017[:26200, :, :]
    y_train_2017 = y_train_2017[:26200, :, :]

    x_train_len_2017 = x_train_len_2017[:26200]
    x_train_before_len_2017 = x_train_before_len_2017[:26200]

    # all train
    all_x_train = np.concatenate((x_train_2016, x_train_2017), axis=0)
    all_y_train = np.concatenate((y_train_2016, y_train_2017), axis=0)

    print('all_x_train:{}'.format(all_x_train.shape))
    print('all_y_train:{}'.format(all_y_train.shape))

    all_x_train = all_x_train[:11400, :, :]
    all_y_train = all_y_train[:11400, :, :]

    print('all_x_train:{}'.format(all_x_train.shape))
    print('all_y_train:{}'.format(all_y_train.shape))

    # all test

    all_x_test = np.concatenate((x_test_2016, x_test_2017), axis=0)
    all_y_test = np.concatenate((y_test_2016, y_test_2017), axis=0)

    print('all_x_test:{}'.format(all_x_test.shape))
    print('all_y_test:{}'.format(all_y_test.shape))

    all_x_test = all_x_test[:1200, :, :]
    all_y_test = all_y_test[:1200, :, :]

    print('all_x_test:{}'.format(all_x_test.shape))
    print('all_y_test:{}'.format(all_y_test.shape))

    return all_x_train, all_y_train, all_x_test, all_y_test, scaler_x, scaler_y


if __name__ == "__main__":
    test_pm25_single_station()
    # train_sampling_params = {
    #     'dim_in': 11,
    #     'output_length': 6,
    #     'min_before': 48,
    #     'max_before': 48,
    #     'min_after': 0,
    #     'max_after': 0,
    #     'file_path': '../data/simplified_PM25.csv'
    # }
    #
    # test_sampling_params = {
    #     'dim_in': 11,
    #     'output_length': 6,
    #     'min_before': 48,
    #     'max_before': 48,
    #     'min_after': 0,
    #     'max_after': 0,
    #     'file_path': '../data/simplified_PM25.csv'
    # }
    #
    # filepath = '../data/simplified_PM25.csv'
    # df = pd.read_csv(filepath, dayfirst=True)
    #
    # df_train, df_test, y, scaler_x, scaler_y = preprocess_df(df)
    #
    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(df_train, train_sampling_params)
    #
    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))
    #
    # x_train, y_train, x_train_len, x_train_before_len = train_test_split_SSIM(x_samples, y_samples, x_len, x_before_len,
    #                                                                           train_sampling_params, SEED)
    #
    # print('x_train:{}'.format(x_train.shape))
    # print('y_train:{}'.format(y_train.shape))
    # print('x_train_len:{}'.format(x_train_len.shape))
    # print('x_train_before_len:{}'.format(x_train_before_len.shape))
    #
    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(df_test, test_sampling_params)
    #
    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))
    #
    # x_test, y_test, x_test_len, x_test_before_len = train_test_split_SSIM(x_samples, y_samples, x_len, x_before_len,
    #                                                                       test_sampling_params, SEED)
    #
    # print('x_test:{}'.format(x_test.shape))
    # print('y_test:{}'.format(y_test.shape))
    # print('x_test_len:{}'.format(x_test_len.shape))
    # print('x_test_before_len:{}'.format(x_test_before_len.shape))
