import time
import os
import math
import numpy as np
import pandas as pd
import random
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
sns.set(style="whitegrid", font_scale=2)

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)


# plots in the paper


# missing data plot

# filepath = r'C:\Users\ZHA244\Dropbox\DigiScape\DigiScape\SensorCloud\Data_Kaggle\resample_missingdata\outer_join\russell_river_east_russell_joined.csv'

# df = pd.read_csv(filepath)
# df.set_index('Timestamp', inplace=True)
# df.drop(columns=['Dayofweek', 'Month'],inplace=True)

# print(df.info())

# df.columns = ['Q', 'Conductivity', 'NO3', 'Temperature','Turbidity', 'Level']
# df = df.loc['2019-02-01T00:00':'2019-03-31T00:00']
# df.replace(0, np.nan, inplace=True)

# print(df.info())


# # msno.matrix(df)
# msno.matrix(df.set_index(pd.period_range(start='2019-02-01', periods=1392, freq='H')) , freq='10D', fontsize=20)
# plt.show()


# temporal variation

filepath = r'C:\Users\ZHA244\Dropbox\DigiScape\DigiScape\SensorCloud\Data_Kaggle\resample_missingdata\outer_join\russell_river_east_russell_joined.csv'

df = pd.read_csv(filepath)
df.set_index('Timestamp', inplace=True)
df.drop(columns=['Dayofweek', 'Month'], inplace=True)

print(df.info())

df.columns = ['Q', 'Conductivity', 'NO3', 'Temperature', 'Turbidity', 'Level']
df = df.loc['2019-02-01T00:00':'2019-02-28T00:00']
df.replace(0, np.nan, inplace=True)

print(df.info())

f, axes = plt.subplots(1, 2, figsize=(20, 20))


ax1 = sns.lineplot(x=df.index, y=df['Level'], ax=axes[0])
ax2 = sns.lineplot(x=df.index, y=df['NO3'], ax=axes[1])
# f.xticks(rotation=45)
# plt.xticks(rotation=45)
axes[0].set_xticklabels(df.index, rotation=45)
axes[1].set_xticklabels(df.index, rotation=45)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(192))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(192))


axes[0].set(xlabel='Date', ylabel='Water Temperature')
axes[1].set(xlabel='Date', ylabel='Nitrate')

plt.tight_layout()
plt.show()
