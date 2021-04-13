import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

pd.set_option('display.expand_frame_repr', False)


def data_together(filepath):
    """
    :param filepath: all the csv files with raw data
    :return:
    """
    csvs = []
    dfs = []

    for subdir, dirs, files in os.walk(filepath):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".csv"):
                csvs.append(filepath)

    for f in csvs:
        temp = pd.read_csv(f)
        # temp['datetime'] = pd.to_datetime(temp['datetime'], dayfirst=True)
        # temp.set_index('datetime', inplace=True)

        # temp.index = pd.to_datetime(temp['Timestamp'], dayfirst=True, utc=True).dt.strftime('%Y-%m-%d %H:%M')
        # temp.drop('Timestamp', axis=1, inplace=True)
        dfs.append(temp)

    return dfs, csvs


def resampleCSV(df, filepath):
    # use day first for GOV download csv
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
    df.set_index('Timestamp', inplace=True)
    newcsv = df.resample('1H').mean()
    # interporation
    # newcsv = newcsv.interpolate(method='linear', axis=0)
    # filedir, name = os.path.split(filepath)
    filename, file_extension = os.path.splitext(filepath)
    # outputcsv = os.path.join(filedir, name + '_resample' + '.csv')
    outputcsv = os.path.join(filename + '_resample' + '.csv')
    newcsv.to_csv(outputcsv, date_format='%Y-%m-%dT%H:%M:%S')


def mergeSameStationList(dflist, filepath):

    newone = pd.merge(dflist[0], dflist[1], on='Timestamp', how='outer')
    remainlist = []
    for df in dflist[2:]:
        remainlist.append(df)

    for i in range(0, len(remainlist)):
        newone = pd.merge(newone, remainlist[i], on='Timestamp', how='outer')

    # newone['Stationname'] = filedir[-1]
    newone['Timestamp'] = pd.to_datetime(newone['Timestamp'])
    newone.set_index('Timestamp', inplace=True)
    newone['Dayofweek'] = newone.index.dayofweek
    newone['Month'] = newone.index.month
    print(newone)

    # output CSV
    filename, file_extension = os.path.splitext(filepath)
    outputcsv = os.path.join(filename + '_joined' + '.csv')
    newone.to_csv(outputcsv, date_format='%Y-%m-%dT%H:%M:%S')


if __name__ == '__main__':

    ### resample data ###

    # datapath = r'C:\Users\ZHA244\Dropbox\DigiScape\DigiScape\SensorCloud\Data_Kaggle\Tully_River_Tully_Gorge_National_Park'
    # alldata,allpaths = data_together(datapath)
    # for i,j in zip(alldata,allpaths):
    #     resampleCSV(i,j)

    ### join variables ###

    datapath = r'C:\Users\ZHA244\Dropbox\DigiScape\DigiScape\SensorCloud\Data_Kaggle\resample_missingdata\Tully_River_Tully_Gorge_National_Park'
    alldata, allpaths = data_together(datapath)
    mergeSameStationList(alldata, datapath)

    # statistics

    # datapath = r'C:\Users\ZHA244\Coding\Pytorch_based\Dual-Head-SSIM\data'
    # alldata,allpaths = data_together(datapath)
    #
    # df_all = pd.concat([alldata[0],alldata[1],alldata[2],alldata[3],alldata[4],], axis=0)
    #
    # print(df_all.describe())

    # # statistics single location
    # datapath = r'C:\Users\ZHA244\Coding\Pytorch_based\Bias_Attention\data\iowa'
    # alldata,allpaths = data_together(datapath)
    #
    # df_all = pd.concat([alldata[0],alldata[1]], axis=0)
    #
    # print(df_all.describe())


# geopandas
# path = r'C:\Users\ZHA244\Downloads\shapefile_site_2018'
# df = geopandas.read_file(path)
# print(df)
# ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
# plt.show()

#
# folder_path = r'C:\Users\ZHA244\Downloads\test'


# ## combin and regroup IOWA water quality data
#
# df_all, _ = data_together(folder_path)
#
# print(len(df_all))
#
#
# # try 1
#
# print(df_all[0].describe())
#
# df_new = df_all[0][df_all[0].site_uid=='WQS0002']
#
# print(df_new.describe())
#
#
#
#
#
#
#
# for i in df_all:
#     print(i.head(1))
