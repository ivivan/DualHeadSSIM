import numpy as np
import pandas as pd
import scipy.stats as stats
import pylab as pl
import matplotlib.pyplot as plt

path = 'data/QLD_nomiss.csv'
df = pd.read_csv(path)
df.set_index('Timestamp', inplace=True)
df.drop(columns=['Dayofweek', 'Month'],inplace=True)

df = df.loc['2019-01-01T00:00':'2019-12-31T23:00']

df[df['Turbidity'] < 0] = 0
df.replace(0, np.nan, inplace=True)





statistic = df.describe().T
print(statistic.loc[:,['min','max','mean','std']])

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





# h1 = df['Temp_degC'].values
# h1.sort()
# fit1 = stats.norm.pdf(h1, np.mean(h1), np.std(h1))  #this is a fitting indeed

# h2 = df['EC_uScm'].values
# h2.sort()
# fit2 = stats.norm.pdf(h2, np.mean(h2), np.std(h2))  #this is a fitting indeed

# h3 = df['pH'].values
# h3.sort()
# fit3 = stats.norm.pdf(h3, np.mean(h3), np.std(h3))  #this is a fitting indeed

# h4 = df['DO_mg'].values
# h4.sort()
# fit4 = stats.norm.pdf(h4, np.mean(h4), np.std(h4))  #this is a fitting indeed

# h5 = df['Turbidity_NTU'].values
# h5.sort()
# fit5 = stats.norm.pdf(h5, np.mean(h5), np.std(h5))  #this is a fitting indeed

# h6 = df['Chloraphylla_ugL'].values
# h6.sort()
# fit6 = stats.norm.pdf(h6, np.mean(h6), np.std(h6))  #this is a fitting indeed





# # Drawing



# # These are the "Tableau 20" colors as RGB.
# tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
# for i in range(len(tableau20)):
#     r, g, b = tableau20[i]
#     tableau20[i] = (r / 255., g / 255., b / 255.)

# fig, ax = plt.subplots(nrows=3, ncols=2)

# plt.subplot(3, 2, 1)
# plt.plot(h1, fit1)
# plt.hist(h1,normed=True)
# plt.gca().set_title('Temperature')
# plt.xlabel('$\u2103$')

# plt.subplot(3, 2, 2)
# plt.plot(h2, fit2)
# plt.hist(h2,normed=True)
# plt.gca().set_title('EC')
# plt.xlabel('uS $cm^{-1}$')

# plt.subplot(3, 2, 3)
# plt.plot(h3, fit3)
# plt.hist(h3,normed=True)
# plt.gca().set_title('pH')
# plt.xlabel('u of pH')

# plt.subplot(3, 2, 4)
# plt.plot(h4, fit4)
# plt.hist(h4,normed=True)
# plt.gca().set_title('DO')
# plt.xlabel('mg $L^{-1}$')

# plt.subplot(3, 2, 5)
# plt.plot(h5, fit5)
# plt.hist(h5,normed=True)
# plt.gca().set_title('Turbidity')
# plt.xlabel('NTU')

# plt.subplot(3, 2, 6)
# plt.plot(h6, fit6)
# plt.hist(h6,normed=True)
# plt.gca().set_title('Chl-a')
# plt.xlabel('$\mu$g $L^{-1}$')

# plt.show()