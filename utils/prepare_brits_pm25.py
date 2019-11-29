import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import math
import os
import time
import json

# from utils.VLSW import pad_all_cases
from VLSW import pad_all_cases

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)


def preprocess_df(df):
    """ The training and testing data are manually selected.
    :param df:  dataframe with raw data
    :return:
    """

    df.set_index('date', inplace=True)

    pm25 = df[['pm2.5']]

    # Standlization, use StandardScaler
    scaler_x = StandardScaler()
    scaler_x.fit(pm25['pm2.5'].values.reshape(-1, 1))
    pm25['pm2.5'] = scaler_x.transform(pm25['pm2.5'].values.reshape(-1, 1))

    df_train = pm25.loc['2/01/2010 0:00':'31/12/2013 23:00'].copy()
    df_test = pm25.loc['1/01/2014 0:00':'31/12/2014 23:00'].copy()

    return df_train, df_test, scaler_x


def train_val_test_generate(dataframe, model_params):
    '''
    :param dataframe: processed dataframe
    :param model_params: for input dim
    :return: train_x, train_y, test_x, test_y with the same length (by padding zero)
    '''

    train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples = pad_all_cases(
        dataframe, dataframe['pm2.5'].values, model_params,
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

    # check and remove samples with NaN (just incase)
    index_list = []
    for index, (x_s, y_s, len_s,
                len_before_s) in enumerate(zip(x, y, x_len, x_before_len)):
        if (np.isnan(x_s).any()) or (np.isnan(y_s).any()):
            index_list.append(index)

    x = np.delete(x, index_list, axis=0)
    y = np.delete(y, index_list, axis=0)
    x_len = np.delete(x_len, index_list, axis=0)
    x_before_len = np.delete(x_before_len, index_list, axis=0)

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
        'dim_in': 1,
        'output_length': 5,
        'min_before': 4,
        'max_before': 4,
        'min_after': 6,
        'max_after': 6,
        'file_path': '../data/simplified_PM25.csv'
    }

    test_sampling_params = {
        'dim_in': 1,
        'output_length': 5,
        'min_before': 4,
        'max_before': 4,
        'min_after': 6,
        'max_after': 6,
        'file_path': '../data/simplified_PM25.csv'
    }

    filepath = 'data/simplified_PM25.csv'

    df = pd.read_csv(filepath, dayfirst=True)

    df_train, df_test, scaler_x = preprocess_df(df)

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_train, train_sampling_params)

    print('X_samples:{}'.format(x_samples.shape))
    print('y_samples:{}'.format(y_samples.shape))

    x_train, y_train, x_train_len, x_train_before_len = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

    print('x_train:{}'.format(x_train.shape))
    print('y_train:{}'.format(y_train.shape))
    print('x_train_len:{}'.format(x_train_len.shape))
    print('x_train_before_len:{}'.format(x_train_before_len.shape))

    x_train = x_train[:944700, :, :]
    y_train = y_train[:944700, :, :]

    x_train_len = x_train_len[:944700]
    x_train_before_len = x_train_before_len[:944700]

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_test, test_sampling_params)

    print('X_samples:{}'.format(x_samples.shape))
    print('y_samples:{}'.format(y_samples.shape))

    x_test, y_test, x_test_len, x_test_before_len = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    print('x_test:{}'.format(x_test.shape))
    print('y_test:{}'.format(y_test.shape))
    print('x_test_len:{}'.format(x_test_len.shape))
    print('x_test_before_len:{}'.format(x_test_before_len.shape))

    x_test = x_test[:6500, :, :]
    y_test = y_test[:6500, :, :]

    x_test_len = x_test_len[:6500]
    x_test_before_len = x_test_before_len[:6500]

    return (x_train, y_train, x_train_len,
            x_train_before_len), (x_test, y_test, x_test_len,
                                  x_test_before_len)


def generate_delta(mask_array):

    deltas = np.zeros(mask_array.shape)
    timetable = np.arange(mask_array.shape[1])

    # fill the delta vectors
    for index, value in np.ndenumerate(mask_array):
        # print(index,value)
        # '''
        # index[0] = row, agg
        # index[1] = col, time
        # '''
        if index[1] == 0:
            deltas[index[0], index[1]] = 0
        elif mask_train[index[0], index[1] - 1] == 0:
            deltas[index[0], index[1]] = timetable[index[1]] - timetable[
                index[1] - 1] + deltas[index[0], index[1] - 1]
        else:
            deltas[index[0], index[1]] = timetable[index[1]] - timetable[
                index[1] - 1]

    return deltas


def generate_masks(x_sample_array, first_split_loc, second_split_loc):

    # split x samples

    sample_list = np.split(x_sample_array, [first_split_loc, second_split_loc],
                           axis=1)

    mask_before = np.ones(sample_list[0].shape)
    mask_middle = np.zeros(sample_list[1].shape)
    mask_after = np.ones(sample_list[2].shape)

    mask_all = np.concatenate((mask_before, mask_middle, mask_after), axis=1)

    return mask_all


def generate_eval_mask(x_sample_array, first_split_loc, second_split_loc):
    # split x samples

    sample_list = np.split(x_sample_array, [first_split_loc, second_split_loc],
                           axis=1)

    mask_before = np.zeros(sample_list[0].shape)
    mask_middle = np.ones(sample_list[1].shape)
    mask_after = np.zeros(sample_list[2].shape)

    mask_all = np.concatenate((mask_before, mask_middle, mask_after), axis=1)

    return mask_all


def generate_eval(x_sample_array, y_sample_array, first_split_loc,
                  second_split_loc):

    # split x samples

    sample_list = np.split(x_sample_array, [first_split_loc, second_split_loc],
                           axis=1)

    value_list = np.concatenate(
        (sample_list[0], y_sample_array, sample_list[2]), axis=1)

    return value_list


def generate_dicts(eval_list, eval_mask_list, value_list, masks_list,
                   delta_list, forward_list, eval_list_bac, eval_mask_list_bac,
                   value_list_bac, masks_list_bac, delta_list_bac,
                   forward_list_bac, train_label):

    size = value_list.shape[0]
    total_samples = []

    for i in range(size):
        line_dict = dict.fromkeys(['forward', 'backward', 'label', 'is_train'])
        temp_dict = dict.fromkeys(
            ['values', 'masks', 'deltas', 'forwards', 'evals', 'eval_masks'])

        # forward
        temp_dict['values'] = value_list[i].flatten().tolist()
        temp_dict['masks'] = masks_list[i].flatten().tolist()
        temp_dict['deltas'] = delta_list[i].flatten().tolist()
        temp_dict['forwards'] = value_list[i].flatten().tolist()
        temp_dict['evals'] = eval_list[i].flatten().tolist()
        temp_dict['eval_masks'] = eval_mask_list[i].flatten().tolist()
        line_dict['forward'] = [temp_dict]
        # backward
        temp_dict['values'] = value_list_bac[i].flatten().tolist()
        temp_dict['masks'] = masks_list_bac[i].flatten().tolist()
        temp_dict['deltas'] = delta_list_bac[i].flatten().tolist()
        temp_dict['forwards'] = value_list_bac[i].flatten().tolist()
        temp_dict['evals'] = eval_list_bac[i].flatten().tolist()
        temp_dict['eval_masks'] = eval_mask_list_bac[i].flatten().tolist()
        line_dict['backward'] = [temp_dict]
        # label
        line_dict['label'] = train_label
        # train/test
        line_dict['is_train'] = train_label
        total_samples.append(line_dict)

    return total_samples


# def write_to_json(pred_dict_list):

if __name__ == "__main__":
    train_sampling_params = {
        'dim_in': 1,
        'output_length': 5,
        'min_before': 5,
        'max_before': 5,
        'min_after': 5,
        'max_after': 5,
        'file_path': '../data/simplified_PM25.csv'
    }

    test_sampling_params = {
        'dim_in': 1,
        'output_length': 5,
        'min_before': 5,
        'max_before': 5,
        'min_after': 5,
        'max_after': 5,
        'file_path': '../data/simplified_PM25.csv'
    }

    filepath = 'data/simplified_PM25.csv'
    df = pd.read_csv(filepath, dayfirst=True)

    df_train, df_test, scaler_x = preprocess_df(df)

    print(df_train.head())

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_train, train_sampling_params)

    print('X_samples:{}'.format(x_samples.shape))
    print('y_samples:{}'.format(y_samples.shape))

    x_train, y_train, x_train_len, x_train_before_len = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

    print('x_train:{}'.format(x_train.shape))
    print('y_train:{}'.format(y_train.shape))
    print('x_train_len:{}'.format(x_train_len.shape))
    print('x_train_before_len:{}'.format(x_train_before_len.shape))

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_test, test_sampling_params)

    print('X_samples:{}'.format(x_samples.shape))
    print('y_samples:{}'.format(y_samples.shape))

    x_test, y_test, x_test_len, x_test_before_len = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    print('x_test:{}'.format(x_test.shape))
    print('y_test:{}'.format(y_test.shape))
    print('x_test_len:{}'.format(x_test_len.shape))
    print('x_test_before_len:{}'.format(x_test_before_len.shape))

    # forward dictionary:

    # mask

    mask_train = generate_masks(
        x_train, train_sampling_params['min_before'],
        train_sampling_params['min_before'] +
        train_sampling_params['output_length'])

    mask_test = generate_masks(
        x_test, test_sampling_params['min_before'],
        test_sampling_params['min_before'] +
        test_sampling_params['output_length'])

    # eval, before elimination

    value_train = generate_eval(
        x_train, y_train, train_sampling_params['min_before'],
        train_sampling_params['min_before'] +
        train_sampling_params['output_length'])

    value_test = generate_eval(
        x_test, y_test, test_sampling_params['min_before'],
        test_sampling_params['min_before'] +
        test_sampling_params['output_length'])

    # eval mask list

    # eval_masks_train = np.ones(mask_train.shape)

    # eval_masks_test = np.ones(mask_test.shape)

    eval_masks_train = generate_eval_mask(
        x_train, train_sampling_params['min_before'],
        train_sampling_params['min_before'] +
        train_sampling_params['output_length'])

    eval_masks_test = generate_eval_mask(
        x_test, test_sampling_params['min_before'],
        test_sampling_params['min_before'] +
        test_sampling_params['output_length'])

    # value list, after elimination

    # x_train
    # x_test

    # generate deltas list

    delta_train = generate_delta(mask_train)
    delta_test = generate_delta(mask_test)

    #-------------------------------------------#
    # backward dictionary:

    # backward the train/test first

    x_train_backward = np.flip(x_train, axis=1)
    y_train_backward = np.flip(y_train, axis=1)
    x_test_backward = np.flip(x_test, axis=1)
    y_test_backward = np.flip(y_test, axis=1)

    # mask

    mask_train_bac = generate_masks(
        x_train_backward, train_sampling_params['min_after'],
        train_sampling_params['min_after'] +
        train_sampling_params['output_length'])

    mask_test_bac = generate_masks(
        x_test_backward, test_sampling_params['min_after'],
        test_sampling_params['min_after'] +
        test_sampling_params['output_length'])

    # eval, before elimination

    value_train_bac = generate_eval(
        x_train_backward, y_train_backward, train_sampling_params['min_after'],
        train_sampling_params['min_after'] +
        train_sampling_params['output_length'])

    value_test_bac = generate_eval(
        x_test_backward, y_test_backward, test_sampling_params['min_after'],
        test_sampling_params['min_after'] +
        test_sampling_params['output_length'])

    # eval mask list

    eval_masks_train_bac = np.ones(mask_train_bac.shape)

    eval_masks_test_bac = np.ones(mask_test_bac.shape)

    eval_masks_train_bac = generate_eval_mask(
        x_train_backward, train_sampling_params['min_after'],
        train_sampling_params['min_after'] +
        train_sampling_params['output_length'])

    eval_masks_test_bac = generate_eval_mask(
        x_test_backward, test_sampling_params['min_after'],
        test_sampling_params['min_after'] +
        test_sampling_params['output_length'])

    # value list, after elimination

    # x_train_backward
    # x_test_backward

    # generate deltas list

    delta_train_bac = generate_delta(mask_train_bac)
    delta_test_bac = generate_delta(mask_test_bac)

    #-------------------------------------------#

    sample_list_train = generate_dicts(value_train, eval_masks_train, x_train,
                                       mask_train, delta_train, x_train,
                                       value_train_bac, eval_masks_train_bac,
                                       x_train_backward, mask_train_bac,
                                       delta_train_bac, x_train_backward, 1)

    sample_list_test = generate_dicts(value_test, eval_masks_test, x_test,
                                      mask_test, delta_test, x_test,
                                      value_test_bac, eval_masks_test_bac,
                                      x_test_backward, mask_test_bac,
                                      delta_test_bac, x_test_backward, 0)

    #-------------------------------------------#

    # sample_list_all = sample_list_train + sample_list_test

    # generate train/test datasets seperately
    with open('test.json', 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in sample_list_test))

    # print('split train/test array')
    # x_test_list = np.split(x_test, [5, 10], axis=1)
    # x_train_list = np.split(x_train, [5, 10], axis=1)

    # mask__test_before = np.ones(x_test_list[0].shape)
    # mask__test_middle = np.zeros(x_test_list[1].shape)
    # mask__test_after = np.ones(x_test_list[2].shape)

    # mask__train_before = np.ones(x_train_list[0].shape)
    # mask__train_middle = np.zeros(x_train_list[1].shape)
    # mask__train_after = np.ones(x_train_list[2].shape)

    # # # concat mask

    # # mask_test_1 = np.concatenate((mask__test_before,mask__test_middle,mask__test_after), axis=1)

    # # mask_train_1 = np.concatenate((mask__train_before,mask__train_middle,mask__train_after), axis=1)

    # # print(mask_train_1.shape)
    # # print(mask_test_1.shape)

    # # if(np.array_equal(mask_train, mask_train_1)):
    # #     print('True')
    # # if(np.array_equal(mask_test,mask_test_1)):
    # #     print('True')

    # # concat train test, before elimination

    # value_test_1 = np.concatenate((x_test_list[0], y_test, x_test_list[2]),
    #                               axis=1)

    # value_train_1 = np.concatenate((x_train_list[0], y_train, x_train_list[2]),
    #                                axis=1)

    # print(value_train.shape)
    # print(value_test.shape)

    # if (np.array_equal(value_test, value_test_1)):
    #     print('True')
    # if (np.array_equal(value_train, value_train_1)):
    #     print('True')

    # print('!!!!!!!!')

    # # train test, after elimination

    # # x_train
    # # x_test

    # # generate deltas list

    # delta_train = generate_delta(mask_train)

    # delta_test = generate_delta(mask_test)

    # print(delta_train.shape)
    # print(delta_train[55])

    # print(delta_test.shape)
    # print(delta_test[55])

    # delta_train = np.zeros(value_train.shape)
    # timetable = np.arange(value_train.shape[1])

    #     # fill the delta vectors
    # for index, value in np.ndenumerate(mask_train):
    #     # print(index,value)
    #     # '''
    #     # index[0] = row, agg
    #     # index[1] = col, time
    #     # '''
    #     if index[1] == 0:
    #         delta_train[index[0], index[1]] = 0
    #     elif mask_train[index[0], index[1] - 1] == 0:
    #         delta_train[index[0], index[1]] = timetable[index[1]] - timetable[
    #             index[1] - 1] + delta_train[index[0], index[1] - 1]
    #     else:
    #         delta_train[index[0], index[1]] = timetable[index[1]] - timetable[
    #             index[1] - 1]

    # print(delta_train.shape)
    # print(delta_train[55])

# ## Different test data

# (x_train, y_train, x_train_len,
#     x_train_before_len), (x_test, y_test, x_test_len,
#                         x_test_before_len) = test_pm25_single_station()

# print('x_train:{}'.format(x_train.shape))
# print('y_train:{}'.format(y_train.shape))
# print('x_test:{}'.format(x_test.shape))
# print('y_test:{}'.format(y_test.shape))

# def generate_dicts():

#     # forward

#     # value list

#     # masks list

#     masking = np.zeros((len(inputdict)-2, grouped_data.ngroups))

#     # deltas list

#     delta = np.zeros((split, size))

#         # fill the delta vectors
#     for index, value in np.ndenumerate(masking):
#         '''
#         index[0] = row, agg
#         index[1] = col, time
#         '''
#         if index[1] == 0:
#             delta[index[0], index[1]] = 0
#         elif masking[index[0], index[1] - 1] == 0:
#             delta[index[0], index[1]] = timetable[index[1]] - timetable[
#                 index[1] - 1] + delta[index[0], index[1] - 1]
#         else:
#             delta[index[0], index[1]] = timetable[index[1]] - timetable[
#                 index[1] - 1]

#     # forwards list

#     # evals list

#     # eval_masks

#     # backword

#     # label
