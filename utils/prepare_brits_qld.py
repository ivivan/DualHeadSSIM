import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
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

    df.set_index('Timestamp', inplace=True)

    # single input/output   DO or Nitrate
    do = df[['Level']]

    ## some variables are not used in training the model, based on the performance evaluation

    # Standlization, use StandardScaler
    scaler_x = MinMaxScaler()
    scaler_x.fit(do['Level'].values.reshape(-1, 1))
    do['Level'] = scaler_x.transform(do['Level'].values.reshape(
        -1, 1))

    # get data from 2014 and 2015
    # 6，7, 8, 9，10 as train; 11 as test


    df_train_one = do.loc['2019-04-01T00:00':'2019-12-31T23:00'].copy()
    df_test_one = do.loc['2019-01-01T00:00':'2019-03-31T23:00'].copy()



    # df_train_one = do.loc['2014-06-01T00:00':'2014-10-31T23:30'].copy()
    # df_train_two = do.loc['2015-06-01T00:00':'2015-10-31T23:30'].copy()

    # df_test_one = do.loc['2014-11-01T00:00':'2014-11-30T23:30'].copy()
    # df_test_two = do.loc['2015-11-01T00:00':'2015-11-30T23:30'].copy()



    # return df_train_one, df_train_two, df_test_one, df_test_two, scaler_x
    return df_train_one, df_test_one, scaler_x

def train_val_test_generate(dataframe, model_params):
    '''
    :param dataframe: processed dataframe
    :param model_params: for input dim
    :return: train_x, train_y, test_x, test_y with the same length (by padding zero)
    '''

    train_val_test_x, train_val_test_y, len_x_samples, len_before_x_samples = pad_all_cases(
        dataframe, dataframe['Level'].values, model_params,
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


def test_qld_single_station():
    train_sampling_params = {
        'dim_in': 1,
        'output_length': 6,
        'min_before': 10,
        'max_before': 10,
        'min_after': 10,
        'max_after': 10,
        'file_path': '../data/QLD_nomiss.csv'
    }

    test_sampling_params = {
        'dim_in': 1,
        'output_length': 6,
        'min_before': 10,
        'max_before': 10,
        'min_after': 10,
        'max_after': 10,
        'file_path': '../data/QLD_nomiss.csv'
    }

    filepath = 'data/QLD_nomiss.csv'

    df = pd.read_csv(filepath)

    # df_train_one, df_train_two, df_train_three, df_train_four, df_test_one, df_test_two, df_test_three, df_test_four, scaler_x = preprocess_df(
    #     df)

    df_train_one, df_test_one, scaler_x = preprocess_df(
        df)

    print('train_preprocess:{}'.format(df_train_one.shape))
    print('test_preprocess:{}'.format(df_test_one.shape))

    # generate train/test samples seperately

    # train one
    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_train_one, train_sampling_params)

    print('X_samples:{}'.format(x_samples.shape))
    print('y_samples:{}'.format(y_samples.shape))

    x_train_one, y_train_one, x_train_len_one, x_train_before_len_one = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

    print('x_train:{}'.format(x_train.shape))
    print('y_train:{}'.format(y_train.shape))
    print('x_train_len:{}'.format(x_train_len.shape))
    print('x_train_before_len:{}'.format(x_train_before_len.shape))

    # # train two
    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
    #     df_train_two, train_sampling_params)

    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))

    # x_train_two, y_train_two, x_train_len_two, x_train_before_len_two = train_test_split_SSIM(
    #     x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

    # print('x_train:{}'.format(x_train.shape))
    # print('y_train:{}'.format(y_train.shape))
    # print('x_train_len:{}'.format(x_train_len.shape))
    # print('x_train_before_len:{}'.format(x_train_before_len.shape))

    # # train three
    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
    #     df_train_three, train_sampling_params)

    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))

    # x_train_three, y_train_three, x_train_len_three, x_train_before_len_three = train_test_split_SSIM(
    #     x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

    # print('x_train:{}'.format(x_train.shape))
    # print('y_train:{}'.format(y_train.shape))
    # print('x_train_len:{}'.format(x_train_len.shape))
    # print('x_train_before_len:{}'.format(x_train_before_len.shape))

    # # train four
    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
    #     df_train_four, train_sampling_params)

    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))

    # x_train_four, y_train_four, x_train_len_four, x_train_before_len_four = train_test_split_SSIM(
    #     x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)

    # print('x_train:{}'.format(x_train.shape))
    # print('y_train:{}'.format(y_train.shape))
    # print('x_train_len:{}'.format(x_train_len.shape))
    # print('x_train_before_len:{}'.format(x_train_before_len.shape))

    # concate all train data

    # all_train_x = np.concatenate(
    #     (x_train_one, x_train_two, x_train_three, x_train_four), axis=0)
    # all_train_y = np.concatenate(
    #     (y_train_one, y_train_two, y_train_three, y_train_four), axis=0)
    all_train_x = x_train_one
    all_train_y = y_train_one
    #------------------------------#

    # test one

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_test_one, test_sampling_params)

    print('X_samples:{}'.format(x_samples.shape))
    print('y_samples:{}'.format(y_samples.shape))

    x_test_one, y_test_one, x_test_len_one, x_test_before_len_one = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    print('x_test:{}'.format(x_test.shape))
    print('y_test:{}'.format(y_test.shape))
    print('x_test_len:{}'.format(x_test_len.shape))
    print('x_test_before_len:{}'.format(x_test_before_len.shape))

    # # test two

    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
    #     df_test_two, test_sampling_params)

    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))

    # x_test_two, y_test_two, x_test_len_two, x_test_before_len_two = train_test_split_SSIM(
    #     x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    # print('x_test:{}'.format(x_test.shape))
    # print('y_test:{}'.format(y_test.shape))
    # print('x_test_len:{}'.format(x_test_len.shape))
    # print('x_test_before_len:{}'.format(x_test_before_len.shape))

    # # test three

    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
    #     df_test_three, test_sampling_params)

    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))

    # x_test_three, y_test_three, x_test_len_three, x_test_before_len_three = train_test_split_SSIM(
    #     x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    # print('x_test:{}'.format(x_test.shape))
    # print('y_test:{}'.format(y_test.shape))
    # print('x_test_len:{}'.format(x_test_len.shape))
    # print('x_test_before_len:{}'.format(x_test_before_len.shape))

    # # test four

    # x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
    #     df_test_four, test_sampling_params)

    # print('X_samples:{}'.format(x_samples.shape))
    # print('y_samples:{}'.format(y_samples.shape))

    # x_test_four, y_test_four, x_test_len_four, x_test_before_len_four = train_test_split_SSIM(
    #     x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)

    # print('x_test:{}'.format(x_test.shape))
    # print('y_test:{}'.format(y_test.shape))
    # print('x_test_len:{}'.format(x_test_len.shape))
    # print('x_test_before_len:{}'.format(x_test_before_len.shape))

    # concate all test data

    # all_test_x = np.concatenate(
    #     (x_test_one, x_test_two, x_test_three, x_test_four), axis=0)
    # all_test_y = np.concatenate(
    #     (y_test_one, y_test_two, y_test_three, y_test_four), axis=0)

    all_test_x = x_test_one
    all_test_y = y_test_one

    print('all_train_x:{}'.format(all_train_x.shape))
    print('all_train_y:{}'.format(all_train_y.shape))
    print('all_test_x:{}'.format(all_test_x.shape))
    print('all_test_y:{}'.format(all_test_y.shape))

    return (all_train_x, all_train_y, x_train_len,
            x_train_before_len), (all_test_x, all_test_y, x_test_len,
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
        'output_length': 6,
        'min_before': 10,
        'max_before': 10,
        'min_after': 10,
        'max_after': 10,
        'file_path': '../data/QLD_nomiss.csv'
    }

    test_sampling_params = {
        'dim_in': 1,
        'output_length': 6,
        'min_before': 10,
        'max_before': 10,
        'min_after': 10,
        'max_after': 10,
        'file_path': '../data/QLD_nomiss.csv'
    }

    # IOWA data

    filepath = 'data/QLD_nomiss.csv'
    df = pd.read_csv(filepath)



    df_train_one, df_test_one, scaler_x = preprocess_df(df)


    print('train_preprocess:{}'.format(df_train_one.shape))
    print('test_preprocess:{}'.format(df_test_one.shape))



    # df_train_one, df_train_two, df_test_one, df_test_two, scaler_x, scaler_y = preprocess_df(
    #     df)

    # generate train/test samples seperately

    # train one
    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_train_one, train_sampling_params)

    x_train_one, y_train_one, x_train_len_one, x_train_before_len_one = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, train_sampling_params, SEED)




    # concate all train data

    x_train = x_train_one
    y_train = y_train_one



    #------------------------------#

    # test one

    x_samples, y_samples, x_len, x_before_len = train_val_test_generate(
        df_test_one, test_sampling_params)

    x_test_one, y_test_one, x_test_len_one, x_test_before_len_one = train_test_split_SSIM(
        x_samples, y_samples, x_len, x_before_len, test_sampling_params, SEED)







    # # concate all test data

    x_test = x_test_one
    y_test = y_test_one


    print('x_train:{}'.format(x_train.shape))
    print('y_train:{}'.format(y_train.shape))
    print('x_test:{}'.format(x_test.shape))
    print('y_test:{}'.format(y_test.shape))

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
    with open('Level_6train0103.json', 'w') as fp:
        fp.write('\n'.join(json.dumps(i) for i in sample_list_train))
    with open('Level_6test0103.json', 'w') as fp:
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
