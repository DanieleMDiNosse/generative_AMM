
#%%

import pickle
import qGaussian
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
sns.set_theme()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

folder = '../data/tmp2'
with open(f'{folder}/usdc_weth_05.pickle', 'rb') as f:
    df = pickle.load(f)
df.index = pd.to_datetime(df.index)

# In this step, I just need the swap data
df = df[ (df.Event == 'Swap_X2Y') | (df.Event == 'Swap_Y2X') ]

# Aggregate the data in df every minute. Keep the last value of each interval
df = df.resample('1T').last()

#%% Good data

start_date = datetime(2022, 1, 1, 00, 00, 00)
end_date = datetime(2022, 1, 8, 00, 00, 00)
x_train = df[(df.index >= start_date) & (df.index < end_date)].price.dropna().values.astype(np.float64)
x_train = 100 * np.log(x_train[1:] / x_train[:-1])

# Count the number of outliers
lq = np.quantile(x_train, 0.25)
uq = np.quantile(x_train, 0.75)
iqr = 1.5 * (uq - lq)
print('There are', 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train), '% outliers in the data')
outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

fitted_values = qGaussian.fit(x_train)
fitted_q, fitted_sigma = fitted_values['q'], fitted_values['sigma']
print(f"Fitted q: {fitted_q}, Fitted sigma: {fitted_sigma}")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.histplot(x_train, ax=ax, stat='probability', label='Data')

num_bins = len(ax.patches) # Number of bins
temp = ax.patches
hist_values = np.histogram(x_train, bins=num_bins, density=True)[0] #Obtain the histogram values
hist_values /= np.sum(hist_values) # Normalize the values
pdf = qGaussian.pdf(np.linspace(np.min(x_train), np.max(x_train), 10000), fitted_q, fitted_sigma)

sns.lineplot(x=np.linspace(np.min(x_train), np.max(x_train), 10000),
             y=pdf*np.max(hist_values)/np.max(pdf),
            ax=ax, color=sns.color_palette()[1], label='Scaled qGaussian PDF')

ax.legend()
plt.show()

plt.scatter(range(len(x_train)), x_train)
plt.show()


#%% Stocks data for comparison

import pickle
with open('../data/stocks_hf.pickle', 'rb') as f:
    stocks_hf = pickle.load(f)[0]

x_train = stocks_hf[
    (stocks_hf.index >= datetime(2020, 6, 1, 00, 00, 00)) &\
        (stocks_hf.index < datetime(2020, 6, 8, 00, 00, 00)) ]['JPM'].values.astype(np.float64)
x_train = 100 * np.log(x_train[1:] / x_train[:-1])
x_train = x_train[ x_train != 0 ]

# Count the number of outliers
lq = np.quantile(x_train, 0.25)
uq = np.quantile(x_train, 0.75)
iqr = 1.5 * (uq - lq)
print('There are', 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train), '% outliers in the data')

fitted_values = qGaussian.fit(x_train)
fitted_q, fitted_sigma = fitted_values['q'], fitted_values['sigma']
print(f"Fitted q: {fitted_q}, Fitted sigma: {fitted_sigma}")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.histplot(x_train, ax=ax, stat='probability', label='Data')

num_bins = len(ax.patches) # Number of bins
temp = ax.patches
hist_values = np.histogram(x_train, bins=num_bins, density=True)[0] #Obtain the histogram values
hist_values /= np.sum(hist_values) # Normalize the values
pdf = qGaussian.pdf(np.linspace(np.min(x_train), np.max(x_train), 10000), fitted_q, fitted_sigma)

sns.lineplot(x=np.linspace(np.min(x_train), np.max(x_train), 10000),
             y=pdf*np.max(hist_values)/np.max(pdf),
            ax=ax, color=sns.color_palette()[1], label='Scaled qGaussian PDF')

ax.legend()
plt.show()

plt.scatter(range(len(x_train)), x_train)
plt.show()

# %%






















#--------------------------------- Fit only q ---------------------------------#
fitted_q, fitted_sigma = qGaussian.fit_only_q(x_train)
print(f"Fitted q: {fitted_q}, Fitted sigma: {fitted_sigma}")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.histplot(x_train, ax=ax, stat='probability', label='Data')

num_bins = len(ax.patches) # Number of bins
hist_values = np.histogram(x_train, bins=num_bins, density=True)[1] #Obtain the histogram values
hist_values /= np.sum(hist_values) # Normalize the values
pdf = qGaussian.pdf(np.linspace(np.min(x_train), np.max(x_train), 10000), fitted_q, fitted_sigma)

sns.lineplot(x=np.linspace(np.min(x_train), np.max(x_train), 10000),
             y=pdf*np.max(hist_values)/np.max(pdf),
            ax=ax, color=sns.color_palette()[1], label='Scaled qGaussian PDF')

ax.legend()
plt.show()
