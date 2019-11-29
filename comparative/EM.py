import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from math import sqrt,fabs
import os,random

import impyute

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)







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

    # get data from 2014 and 2015
    # 6，7, 8, 9，10 as train; 11 as test

    df_train_one = df.loc['2014-06-01T00:00':'2014-10-31T23:30'].copy()
    df_train_two = df.loc['2015-06-01T00:00':'2015-10-31T23:30'].copy()

    df_test_one = df.loc['2014-11-01T00:00':'2014-11-30T23:30'].copy()
    df_test_two = df.loc['2015-11-10T00:00':'2015-11-30T23:30'].copy()

    return df_train_one, df_train_two, df_test_one, df_test_two, scaler_x

    # scaler_y = StandardScaler()
    # scaler_y.fit(tw)
    # y_all = scaler_y.transform(tw)



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

    filepath = './data/45_joined.csv'

    df = pd.read_csv(filepath, dayfirst=True)

    df_train_one, df_train_two, df_test_one, df_test_two, scaler_x = preprocess_df(
        df)
    

    obs = []
    predictions = []


    for i in range(0,df_test_one.shape[0]-30):

        complete_df = df_test_one.iloc[i:i+30].copy()

        temp_y = complete_df.iloc[12:18,4].values


        incomplete_df = complete_df.copy()
        incomplete_df.iloc[12:18,4] = np.nan

        # KNN
        filled  = impyute.fast_knn(incomplete_df)
        ## expectation maximization
        # filled = impyute.imputation.cs.em(incomplete_df, loops=50, dtype='cont')

        obs.append(temp_y.tolist())
        predictions.append(filled.iloc[12:18,4].values.tolist())


    for i in range(0,df_test_two.shape[0]-30):


        complete_df = df_test_two.iloc[i:i+30].copy()



        temp_y = complete_df.iloc[12:18,4].values


        incomplete_df = complete_df.copy()
        incomplete_df.iloc[12:18,4] = np.nan

        # KNN
        filled  = impyute.fast_knn(incomplete_df)

        # # expectation maximization
        # filled = impyute.imputation.cs.em(incomplete_df, loops=50, dtype='cont')

        obs.append(temp_y.tolist())
        predictions.append(filled.iloc[12:18,4].values.tolist())




    # # Default Parameters
    # model_params = {
    #     'TIMESTEPS': 0,
    #     'N_FEATURES': 1,
    #     'BATCH_SIZE': 10,
    #     'dim_in': 17,
    #     'dim_out': 1,
    #     'max_time': 7,
    #     'N_EPOCHS': 50,
    #     'units': 21,
    #     'dropout': 0.5,
    #     'depth': 1,
    #     'k_fold':5,
    #     'output_length':7
    # }

    # datapath = r'C:\Users\ZHA244\Coding\QLD\newQGdata\final'
    # outputdir_each = r'C:\Users\ZHA244\Coding\QLD\newQGdata\model'
    # index_save_path = './index.txt'




















    



    # predictions = list()
    # obs = list()
    # #
    # for i in range(31-model_params['output_length']+1):
    # # for i in range(1):
    #     alldata, allpaths = data_together(datapath)
    #     complete_matrix, incomplete_matrix, Origin_N = preprocessdf(alldata[0],2017,8,1,model_params['output_length']-1,i)

    #     input_matrix = incomplete_matrix.as_matrix()

    #     ##impulation

    #     # ## MatrixFactorization
    #     # solver = MatrixFactorization(
    #     #     learning_rate=0.01,
    #     #     rank=3,
    #     #     l2_penalty=0,
    #     #     min_improvement=1e-6)
    #     # X_filled_knn = solver.complete(input_matrix)

    #     # # fast_knn
    #     # filled = impyute.imputation.cs.fast_knn(input_matrix)

    #     # Multivariate Imputation by Chained Equations
    #     # filled = impyute.imputation.cs.mice(input_matrix)

    #      # expectation maximization
    #     filled = impyute.imputation.cs.em(input_matrix, loops=50, dtype='cont')

    #     # last observation carried forward
    #     # filled = impyute.imputation.ts.locf(input_matrix, axis=1)

    #     pred = filled[212+i:212+i+model_params['output_length']][:,4]

    #     obs.append(Origin_N.tolist())
    #     predictions.append(pred.tolist())

    #     # fig, ax = plt.subplots(1, 1)
    #     # ax.plot(Origin_N.tolist())
    #     # ax.plot(pred.tolist())
    #     # plt.show(block=False)  # That's important



    #     # ## KNN
    #     # X_filled_knn = KNN(k=3).fit_transform(input_matrix)
    #     # print(X_filled_knn[212:220])

    #     # ## NuclearNormMinimization
    #     # solver = NuclearNormMinimization(require_symmetric_solution=False)
    #     # completed = solver.complete(input_matrix)
    #     # # X_filled_knn = NuclearNormMinimization().complete(incomplete_matrix)
    #     # print(completed[212:220])

    #     # ## SimilarityWeightedAveraging
    #     # solver = SimilarityWeightedAveraging()
    #     # X_filled_knn = solver.complete(input_matrix)
    #     # print(X_filled_knn[212:220])

    #     # # print(type(input_matrix))
    #     # #
    #     # biscaler = BiScaler()
    #     # softImpute = SoftImpute()
    #     # # rescale both rows and columns to have zero mean and unit variance
    #     # # X_incomplete_normalized = biscaler.fit_transform(input_matrix)
    #     #
    #     # X_filled_softimpute_normalized = softImpute.complete(input_matrix)
    #     # # X_filled_softimpute = biscaler.inverse_transform(X_filled_softimpute_normalized)
    #     # print(X_filled_softimpute_normalized[212:220])



    #     ##  expectation maximization
    #     # filled = impyute.imputations.cs.em(input_matrix, loops=50, dtype='cont')
    #     # print(filled[212:220])

    #     # print(input_matrix.shape)
    #     # filled  = impyute.fast_knn(input_matrix)
    #     # print(filled)

    # for i in range(30 - model_params['output_length'] + 1):
    #     # for i in range(1):
    #     alldata, allpaths = data_together(datapath)
    #     complete_matrix, incomplete_matrix, Origin_N = preprocessdf(alldata[0], 2018, 4, 1,
    #                                                                 model_params['output_length'] - 1, i)

    #     input_matrix = incomplete_matrix.as_matrix()
    #     ##impulation

    #     #  expectation maximization
    #     filled = impyute.imputation.cs.em(input_matrix, loops=50, dtype='cont')


    #     pred = filled[455 + i:455 + i + model_params['output_length']][:, 4]

    #     obs.append(Origin_N.tolist())
    #     predictions.append(pred.tolist())




    # print(len(obs))
    # print(len(predictions))

    # print(obs)
    # print(predictions)

    # # for t in range(size):
    # #     history = all_cases[t][0:model_params['max_time']]
    # #     seasonal_model = sarimax.SARIMAX(history, order=(1, 0, 1), seasonal_order=(1, 0, 0, 7),
    # #                                      enforce_stationarity=False, enforce_invertibility=False)
    # #     res = seasonal_model.fit(disp=0)
    # #     output = res.forecast(steps=model_params['output_length'])
    # #     predictions.append(output.tolist())
    # #     obs.append(all_cases[t][model_params['max_time']:model_params['max_time']+model_params['output_length']].tolist())


    predictions = [item for sublist in predictions for item in sublist]
    obs = [item for sublist in obs for item in sublist]


    index_list = []
    for index, (x_s, y_s) in enumerate(zip(predictions, obs)):
        if (np.isnan(x_s).any()) or (np.isnan(y_s).any()):
            index_list.append(index)

    predictions = np.delete(predictions, index_list, axis=0)
    obs = np.delete(obs, index_list, axis=0)



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
    mape = mean_absolute_percentage_error(obs, predictions)
    print("MAPE (sklearn):{0:f}".format(mape))
    print("---------")
    smape = symmetric_mean_absolute_percentage_error(np.array(obs), np.array(predictions))
    print("SMAPE (sklearn):{0}".format(smape))