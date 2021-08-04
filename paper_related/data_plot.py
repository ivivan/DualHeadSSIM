import numpy as np
import pandas as pd
import scipy.stats as stats
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns


path = 'data/QLD_nomiss.csv'
df = pd.read_csv(path)
df.drop(['Timestamp','Dayofweek','Month'], axis=1, inplace=True)
statistic = df.describe().T
print(statistic.loc[:,['mean','std']])

# h = df['Temp_degC'].values
# h1 = df['EC_uScm'].values
# h.sort()
# h1.sort()
# print(h)

# h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
#      187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
#      161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180])  #sorted

# fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
# pl.plot(h,fit,'*')
# pl.hist(h,normed=True)      #use this to draw histogram of your data


# fit1 = stats.norm.pdf(h1, np.mean(h1), np.std(h1))  #this is a fitting indeed
# # pl.plot(h1,fit1,'.')
# # pl.hist(h1,normed=True)
# plt.plot(h1, fit1)
# plt.hist(h1,normed=True)
# pl.show()





h1 = df['Q'].values
h1.sort()
fit1 = stats.norm.pdf(h1, np.mean(h1), np.std(h1))  #this is a fitting indeed

h2 = df['Conductivity'].values
h2.sort()
fit2 = stats.norm.pdf(h2, np.mean(h2), np.std(h2))  #this is a fitting indeed

h3 = df['NO3'].values
h3.sort()
fit3 = stats.norm.pdf(h3, np.mean(h3), np.std(h3))  #this is a fitting indeed

h4 = df['Temp'].values
h4.sort()
fit4 = stats.norm.pdf(h4, np.mean(h4), np.std(h4))  #this is a fitting indeed

h5 = df['Turbidity'].values
h5.sort()
fit5 = stats.norm.pdf(h5, np.mean(h5), np.std(h5))  #this is a fitting indeed

h6 = df['Level'].values
h6.sort()
fit6 = stats.norm.pdf(h6, np.mean(h6), np.std(h6))  #this is a fitting indeed





# Drawing



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

fig, ax = plt.subplots(nrows=3, ncols=2)

# plt.subplot(3, 2, 1)
# plt.plot(h1, fit1)
# plt.hist(h1,density=True)
# plt.gca().set_title('Temperature')
# plt.xlabel('$\u2103$')

sns.distplot(df['Q'], hist=True, kde=True, 
             bins=int(180/5), color = tableau20[2], 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1},ax=ax[0, 0])

ax[0, 0].set(xlabel='Water charge')

sns.distplot(df['Conductivity'], hist=True, kde=True, 
             bins=int(180/5), color = tableau20[4], 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1},ax=ax[0, 1])

sns.distplot(df['NO3'], hist=True, kde=True, 
             bins=int(180/5), color = tableau20[6], 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1},ax=ax[1, 0])

ax[1, 0].set(xlabel='Nitrate')

sns.distplot(df['Temp'], hist=True, kde=True, 
             bins=int(180/5), color = tableau20[8], 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1},ax=ax[1, 1])

ax[1, 1].set(xlabel='Water temperature')

sns.distplot(df['Turbidity'], hist=True, kde=True, 
             bins=int(180/5), color = tableau20[10], 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1},ax=ax[2, 0])

sns.distplot(df['Level'], hist=True, kde=True, 
             bins=int(180/5), color = tableau20[12], 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1},ax=ax[2, 1])

ax[2, 1].set(xlabel='Water level')

plt.show()