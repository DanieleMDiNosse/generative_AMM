
#%%
#appurl.io/VqdosOt3na


import pickle
import qGaussian
import qlogNormal
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

folder = '../data'
with open(f'{folder}/usdc_weth_005.pickle', 'rb') as f:
    df = pickle.load(f)
df.index = pd.to_datetime(df.index)

# To avoid duplicates
potential_duplicated = df[ df.duplicated() ].block_number.unique()
sure_duplicated = list()
for block in potential_duplicated:
    temp_df = df[ df.block_number==block ]
    if (temp_df.duplicated()).sum()*2 == len(temp_df):
        sure_duplicated.append(block)
df = df[~ (df['block_number'].isin(sure_duplicated) & df.duplicated())]

# In this step, I just need the swap data
df = df[ (df.Event == 'Swap_X2Y') | (df.Event == 'Swap_Y2X') ]

# Aggregate the data in df every minute. Keep the last value of each interval
df = df.resample('1T').last()

#%% USDC-WETH data. Daily

q_values_list = list()
sigma_list = list()
mean_list = list()
outliers_pct_list = list()
total_obs_list = list()
error_list = list()

for day in tqdm(pd.date_range(datetime(2021, 7, 1, 00, 00, 00),
                              datetime(2024, 5, 1, 00, 00, 00)),
                desc='Iterating over the days'):
    print(day)
    start_date = datetime(day.year, day.month, day.day, 00, 00, 00)
    end_day = day + pd.Timedelta(days=1)
    end_date = datetime(end_day.year, end_day.month, end_day.day, 00, 00, 00)
    x_train = df[(df.index >= start_date) & (df.index < end_date)].price.dropna().values.astype(np.float64)
    x_train = 100 * np.log(x_train[1:] / x_train[:-1])
    x_train = x_train[ x_train != 0] # Remove the zeros
    total_obs_list.append(len(x_train)) # Add the total number of observations

    mean_val = np.mean(x_train)
    mean_list.append(mean_val) # Add the mean value
    x_train = x_train - mean_val # Remove the mean from the data

    # Count the number of outliers
    lq = np.quantile(x_train, 0.25)
    uq = np.quantile(x_train, 0.75)
    iqr = 1.5 * (uq - lq)
    outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
    outliers_pct_list.append(outliers_pct) # Add the percentage of outliers
    print('There are', outliers_pct, '% outliers in the data')
    outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

    try:
        fitted_values = qGaussian.fit(x_train)
        fitted_q, fitted_sigma = fitted_values['q'], fitted_values['sigma']
        print(f"Fitted q: {fitted_q}, Fitted sigma: {fitted_sigma}")
        q_values_list.append(fitted_q) # Add the fitted q value
        sigma_list.append(fitted_sigma) # Add the fitted sigma value
    except:
        print('Error in the fitting process!')
        error_list.append(day)
    print('\n')

normal_crypto_res = [total_obs_list, mean_list, outliers_pct_list,
                     q_values_list, sigma_list, error_list]

#%% JPM data. Daily

import pickle
with open('../data/stocks_hf.pickle', 'rb') as f:
    stocks_hf = pickle.load(f)[0]

q_values_list = list()
sigma_list = list()
mean_list = list()
outliers_pct_list = list()
total_obs_list = list()
error_list = list()

for day in tqdm(pd.date_range(datetime(2020, 2, 1, 00, 00, 00),
                              datetime(2021, 7, 1, 00, 00, 00)),
                desc='Iterating over the days'):
    print(day)
    start_date = datetime(day.year, day.month, day.day, 00, 00, 00)
    end_day = day + pd.Timedelta(days=1)
    end_date = datetime(end_day.year, end_day.month, end_day.day, 00, 00, 00)
    x_train = stocks_hf[(stocks_hf.index >= start_date) &\
                        (stocks_hf.index < end_date)].JPM.values.astype(np.float64)
    x_train = 100 * np.log(x_train[1:] / x_train[:-1])
    x_train = x_train[ x_train != 0] # Remove the zeros
    if len(x_train) > 0:
        total_obs_list.append(len(x_train)) # Add the total number of observations

        mean_val = np.mean(x_train)
        mean_list.append(mean_val) # Add the mean value
        x_train = x_train - mean_val # Remove the mean from the data

        # Count the number of outliers
        lq = np.quantile(x_train, 0.25)
        uq = np.quantile(x_train, 0.75)
        iqr = 1.5 * (uq - lq)
        outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
        outliers_pct_list.append(outliers_pct) # Add the percentage of outliers
        print('There are', outliers_pct, '% outliers in the data')
        outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

        try:
            fitted_values = qGaussian.fit(x_train)
            fitted_q, fitted_sigma = fitted_values['q'], fitted_values['sigma']
            print(f"Fitted q: {fitted_q}, Fitted sigma: {fitted_sigma}")
            q_values_list.append(fitted_q) # Add the fitted q value
            sigma_list.append(fitted_sigma) # Add the fitted sigma value
        except:
            print('Error in the fitting process!')
            error_list.append(day)
        print('\n')

stock_res = [total_obs_list, mean_list, outliers_pct_list,
                     q_values_list, sigma_list, error_list]

#%% Comparison scatterplots

for n_to_plot, to_plot in enumerate(['Series Length', 'Mean Value', 'Outliers %', 'q Value', 'sigma Value']):
    fig, ax = plt.subplots(1,1)
    sns.scatterplot(x=range(len(normal_crypto_res[n_to_plot])),
                    y=normal_crypto_res[n_to_plot], label='USDC-WETH')
    sns.scatterplot(x=range(len(stock_res[n_to_plot])),
                    y=stock_res[n_to_plot], label='JPM')
    ax.axhline(np.mean(normal_crypto_res[n_to_plot]), color=sns.color_palette()[0],
               label='Mean USDC-WETH')
    ax.axhline(np.mean(stock_res[n_to_plot]), color=sns.color_palette()[1],
               label='Mean JPM')
    if to_plot in ['Outliers %', 'q Value']:
        ax.axhline(np.mean(stock_res[n_to_plot]) + np.std(stock_res[n_to_plot]),
                   linestyle='--', color=sns.color_palette()[1], label='Mean+Std JPM')
    plt.title(to_plot)
    plt.legend(bbox_to_anchor=(1, 1.05))
    plt.show()

#%% The inuition is that Outliers % and q Value for USDC-WETH are higher than for stocks. Verify

import pickle
with open('../data/stocks_hf.pickle', 'rb') as f:
    stocks_hf = pickle.load(f)[0]

stocks_res = dict()
for asset in tqdm(stocks_hf.columns, desc='Iterating over the stocks'):
    q_values_list = list()
    sigma_list = list()
    mean_list = list()
    outliers_pct_list = list()
    total_obs_list = list()
    error_list = list()

    for day in pd.date_range(datetime(2020, 2, 1, 00, 00, 00),
                                datetime(2021, 7, 1, 00, 00, 00)):
        start_date = datetime(day.year, day.month, day.day, 00, 00, 00)
        end_day = day + pd.Timedelta(days=1)
        end_date = datetime(end_day.year, end_day.month, end_day.day, 00, 00, 00)
        x_train = stocks_hf[(stocks_hf.index >= start_date) &\
                            (stocks_hf.index < end_date)][asset].values.astype(np.float64)
        x_train = 100 * np.log(x_train[1:] / x_train[:-1])
        x_train = x_train[ x_train != 0] # Remove the zeros
        if len(x_train) > 0:
            total_obs_list.append(len(x_train)) # Add the total number of observations

            mean_val = np.mean(x_train)
            mean_list.append(mean_val) # Add the mean value
            x_train = x_train - mean_val # Remove the mean from the data

            # Count the number of outliers
            lq = np.quantile(x_train, 0.25)
            uq = np.quantile(x_train, 0.75)
            iqr = 1.5 * (uq - lq)
            outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
            outliers_pct_list.append(outliers_pct) # Add the percentage of outliers
            outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

            try:
                fitted_values = qGaussian.fit(x_train)
                fitted_q, fitted_sigma = fitted_values['q'], fitted_values['sigma']
                q_values_list.append(fitted_q) # Add the fitted q value
                sigma_list.append(fitted_sigma) # Add the fitted sigma value
            except:
                error_list.append(day)

    stocks_res[asset] = [outliers_pct_list, q_values_list]

fig, ax = plt.subplots(len(stocks_res.keys()),2, figsize=(16,40))
for n_asset, asset in enumerate(stocks_res.keys()):
    for n_to_plot, to_plot in enumerate(['Outliers %', 'q Value']):
        sns.scatterplot(x=range(len(normal_crypto_res[2+n_to_plot])),
                        y=normal_crypto_res[2+n_to_plot], label='USDC-WETH', ax=ax[n_asset, n_to_plot])
        sns.scatterplot(x=range(len(stocks_res[asset][n_to_plot])),
                        y=stocks_res[asset][n_to_plot], label=asset, ax=ax[n_asset, n_to_plot])
        ax[n_asset, n_to_plot].axhline(np.mean(normal_crypto_res[2+n_to_plot]), color=sns.color_palette()[0],
                label='Mean USDC-WETH')
        ax[n_asset, n_to_plot].axhline(np.mean(stocks_res[asset][n_to_plot]), color=sns.color_palette()[1],
                label='Mean '+asset)
        ax[n_asset, n_to_plot].axhline(np.mean(stocks_res[asset][n_to_plot]) + np.std(stocks_res[asset][n_to_plot]),
                    linestyle='--', color=sns.color_palette()[1], label='Mean+Std '+asset)
        ax[n_asset, n_to_plot].set_title(asset+' - '+to_plot)
    ax[n_asset, 1].legend(bbox_to_anchor=(1, 1.05))
plt.tight_layout()
plt.savefig('../figures/Outlier3q_stocks_vs_crypto.png', dpi=300)
plt.show()

#%% Now, compare different USDC-WETH pools

fee_1 = True

with open(f'{folder}/usdc_weth_03.pickle', 'rb') as f:
    df03 = pickle.load(f)
df03.index = pd.to_datetime(df03.index)
df03 = df03[ (df03.Event == 'Swap_X2Y') | (df03.Event == 'Swap_Y2X') ]
df03 = df03.resample('1T').last()

with open(f'{folder}/usdc_weth_1.pickle', 'rb') as f:
    df1 = pickle.load(f)
df1.index = pd.to_datetime(df1.index)
df1 = df1[ (df1.Event == 'Swap_X2Y') | (df1.Event == 'Swap_Y2X') ]
df1 = df1.resample('1T').last()

normal_crypto_res03_1 = list()
for df_val in [df03, df1]:
    q_values_list = list()
    sigma_list = list()
    mean_list = list()
    outliers_pct_list = list()
    total_obs_list = list()
    error_list = list()
    
    for day in tqdm(pd.date_range(datetime(2021, 7, 1, 00, 00, 00),
                                datetime(2022, 12, 1, 00, 00, 00)),
                    desc='Iterating over the days'):
        start_date = datetime(day.year, day.month, day.day, 00, 00, 00)
        end_day = day + pd.Timedelta(days=1)
        end_date = datetime(end_day.year, end_day.month, end_day.day, 00, 00, 00)
        x_train = df_val[(df_val.index >= start_date) & (df_val.index < end_date)].price.dropna().values.astype(np.float64)
        x_train = 100 * np.log(x_train[1:] / x_train[:-1])
        x_train = x_train[ x_train != 0] # Remove the zeros
        total_obs_list.append(len(x_train)) # Add the total number of observations

        if len(x_train) > 0:
            mean_val = np.mean(x_train)
            mean_list.append(mean_val) # Add the mean value
            x_train = x_train - mean_val # Remove the mean from the data

            # Count the number of outliers
            lq = np.quantile(x_train, 0.25)
            uq = np.quantile(x_train, 0.75)
            iqr = 1.5 * (uq - lq)
            outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
            outliers_pct_list.append(outliers_pct) # Add the percentage of outliers
            outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

            try:
                fitted_values = qGaussian.fit(x_train)
                fitted_q, fitted_sigma = fitted_values['q'], fitted_values['sigma']
                q_values_list.append(fitted_q) # Add the fitted q value
                sigma_list.append(fitted_sigma) # Add the fitted sigma value
            except:
                error_list.append(day)

    normal_crypto_res03_1.append([total_obs_list, mean_list, outliers_pct_list,
                                  q_values_list, sigma_list, error_list])

normal_crypto_res03 = normal_crypto_res03_1[0]
normal_crypto_res1 = normal_crypto_res03_1[1]

# Comparative scatterplots
for n_to_plot, to_plot in enumerate(['Series Length', 'Mean Value', 'Outliers %', 'q Value', 'sigma Value']):
    fig, ax = plt.subplots(1,1)
    sns.scatterplot(x=range(len(normal_crypto_res[n_to_plot])),
                    y=normal_crypto_res[n_to_plot], label='fee=0.05%')
    sns.scatterplot(x=range(len(normal_crypto_res03[n_to_plot])),
                    y=normal_crypto_res03[n_to_plot], label='fee=0.3%')
    ax.axhline(np.mean(normal_crypto_res[n_to_plot]), color=sns.color_palette()[0],
               label='Mean fee=0.05%')
    ax.axhline(np.mean(normal_crypto_res03[n_to_plot]), color=sns.color_palette()[1],
               label='Mean fee=0.3%')
    if fee_1:
        sns.scatterplot(x=range(len(normal_crypto_res1[n_to_plot])),
                        y=normal_crypto_res1[n_to_plot], label='fee=1%')
        ax.axhline(np.mean(normal_crypto_res1[n_to_plot]), color=sns.color_palette()[2],
                label='Mean fee=1%')
    plt.title(to_plot+' USDC-WETH')
    plt.legend(bbox_to_anchor=(1, 1.05))
    plt.show()

#%% Comparison normal vs stable coins

with open(f'{folder}/usdc_usdt_001.pickle', 'rb') as f:
    df_s = pickle.load(f)
df_s.index = pd.to_datetime(df_s.index)
df_s = df_s[ (df_s.Event == 'Swap_X2Y') | (df_s.Event == 'Swap_Y2X') ]
df_s = df_s.resample('1T').last()

q_values_list = list()
sigma_list = list()
mean_list = list()
outliers_pct_list = list()
total_obs_list = list()
error_list = list()

for day in tqdm(pd.date_range(datetime(2021, 7, 1, 00, 00, 00),
                            datetime(2022, 12, 1, 00, 00, 00)),
                desc='Iterating over the days'):
    start_date = datetime(day.year, day.month, day.day, 00, 00, 00)
    end_day = day + pd.Timedelta(days=1)
    end_date = datetime(end_day.year, end_day.month, end_day.day, 00, 00, 00)
    x_train = df_s[(df_s.index >= start_date) & (df_s.index < end_date)].price.dropna().values.astype(np.float64)
    x_train = 100 * np.log(x_train[1:] / x_train[:-1])
    x_train = x_train[ x_train != 0] # Remove the zeros
    total_obs_list.append(len(x_train)) # Add the total number of observations

    if len(x_train) > 0:
        mean_val = np.mean(x_train)
        mean_list.append(mean_val) # Add the mean value
        x_train = x_train - mean_val # Remove the mean from the data

        # Count the number of outliers
        lq = np.quantile(x_train, 0.25)
        uq = np.quantile(x_train, 0.75)
        iqr = 1.5 * (uq - lq)
        outliers_pct = 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train)
        outliers_pct_list.append(outliers_pct) # Add the percentage of outliers
        outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

        try:
            fitted_values = qGaussian.fit(x_train)
            fitted_q, fitted_sigma = fitted_values['q'], fitted_values['sigma']
            q_values_list.append(fitted_q) # Add the fitted q value
            sigma_list.append(fitted_sigma) # Add the fitted sigma value
        except:
            error_list.append(day)

stable_crypto_res = [total_obs_list, mean_list, outliers_pct_list,
                                q_values_list, sigma_list, error_list]

# Comparative scatterplots
for n_to_plot, to_plot in enumerate(['Series Length', 'Mean Value', 'Outliers %', 'q Value', 'sigma Value']):
    fig, ax = plt.subplots(1,1)
    sns.scatterplot(x=range(len(normal_crypto_res[n_to_plot])),
                    y=normal_crypto_res[n_to_plot], label='Normal pair')
    sns.scatterplot(x=range(len(stable_crypto_res[n_to_plot])),
                    y=stable_crypto_res[n_to_plot], label='Stable pair')
    ax.axhline(np.mean(normal_crypto_res[n_to_plot]), color=sns.color_palette()[0],
               label='Normal pair')
    ax.axhline(np.mean(stable_crypto_res[n_to_plot]), color=sns.color_palette()[1],
               label='Stable pair')
    # if to_plot in ['Outliers %', 'q Value']:
    #     ax.axhline(np.mean(normal_crypto_res03[n_to_plot]) + np.std(normal_crypto_res03[n_to_plot]),
    #                linestyle='--', color=sns.color_palette()[1], label='Mean+Std fee=0.3%')
    plt.title(to_plot+' USDC-WETH vs USDC-USDT')
    plt.legend(bbox_to_anchor=(1, 1.05))
    plt.show()

#%% USDC-WETH 1 year of data. Plot q against minute aggregation

folder = '../data/tmp3'
with open(f'{folder}/usdc_weth_005.pickle', 'rb') as f:
    df_base = pickle.load(f)
df_base.index = pd.to_datetime(df_base.index)

# To avoid duplicates
potential_duplicated = df_base[ df_base.duplicated() ].block_number.unique()
sure_duplicated = list()
for block in potential_duplicated:
    temp_df = df_base[ df_base.block_number==block ]
    if (temp_df.duplicated()).sum()*2 == len(temp_df):
        sure_duplicated.append(block)
df_base = df_base[~ (df_base['block_number'].isin(sure_duplicated) & df_base.duplicated())]

# In this step, I just need the swap data
df_base = df_base[ (df_base.Event == 'Swap_X2Y') | (df_base.Event == 'Swap_Y2X') ]

q_values, sigma_values = list(), list()
for min_agg in range(1,11):
    print('Work with minute aggregation:', min_agg)
    # Aggregate the data in df every minute. Keep the last value of each interval
    df = df.resample(f'{min_agg}T').last()

    start_date = datetime(2021, 7, 1, 00, 00, 00)
    end_date = datetime(2022, 7, 1, 00, 00, 00)
    x_train = df[(df.index >= start_date) & (df.index < end_date)].price.dropna().values.astype(np.float64)
    x_train = 100 * np.log(x_train[1:] / x_train[:-1])
    x_train = x_train[ x_train != 0] # Remove the zeros

    mean_val = np.mean(x_train)
    x_train = x_train - mean_val # Remove the mean from the data

    fitted_values = qGaussian.fit(x_train, n_it=100)
    fitted_q, fitted_sigma = fitted_values['q'], fitted_values['sigma']
    print(f"Fitted q: {fitted_q}, Fitted sigma: {fitted_sigma}")
    q_values.append(fitted_q); sigma_values.append(fitted_sigma)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.lineplot(x=range(1,11), y=q_values, ax=ax[0])
ax[0].set_title('q Value')
ax[0].set_xlabel('Minute Aggregation')
ax[0].set_ylabel('q Value')

sns.lineplot(x=range(1,11), y=sigma_values, ax=ax[1])
ax[1].set_title('Sigma Value')
ax[1].set_xlabel('Minute Aggregation')
ax[1].set_ylabel('Sigma Value')

plt.show()

#%% Volume Analysis

def my_power_law(x, alpha):
    return alpha * x ** (-alpha - 1)

def qexp_power_law_pdf_standard_old(x, q, alpha, start_int=0):
    from scipy.integrate import quad
    # print(q)
    # print(alpha)
    try:
        C = 1 / quad(
            lambda z: qGaussian.q_exponential(-z, q) * my_power_law(z, alpha),
            start_int, start_int+10)[0]
    except:
        C = 1 / quad(
            lambda z: qGaussian.q_exponential(-z, q) * my_power_law(z, alpha),
            start_int, start_int+1)[0]
    return C * qGaussian.q_exponential(-x, q) * my_power_law(x, alpha)

def qexp_power_law_pdf_standard(x, q, alpha, start_int=0):
    from scipy.integrate import quad
    # print(q)
    # print(alpha)
    if q < 1:
        # C = 1 / quad(
        #     lambda z: qGaussian.q_exponential(-z, q) * my_power_law(z, alpha),
        #     0, 1/(1-q))[0]
        try:
            C = 1 / quad(
                lambda z: qGaussian.q_exponential(-z, q) * my_power_law(z, alpha),
                -10, 1/(1-q))[0]
        except:
            C = 1 / quad(
                lambda z: qGaussian.q_exponential(-z, q) * my_power_law(z, alpha),
                -1, 1/(1-q))[0]
    else:
        try:
            C = 1 / quad(
                lambda z: qGaussian.q_exponential(-z, q) * my_power_law(z, alpha),
                1/(1-q), 10)[0]
        except:
            try:
                C = 1 / quad(
                    lambda z: qGaussian.q_exponential(-z, q) * my_power_law(z, alpha),
                    1/(1-q), 1)[0]
            except:
                z = np.linspace(0,10,1001)[1:]
                delta_x = 10/1000
                C = 1 / np.sum(
                    qGaussian.q_exponential(-z, q) * my_power_law(z, alpha) * delta_x)
    return C * qGaussian.q_exponential(-x, q) * my_power_law(x, alpha)

def qexp_power_law_pdf(x, q, alpha, loc=0, scale=1):
    return qexp_power_law_pdf_standard((x-loc)/scale, q, alpha, start_int=-loc/scale) / scale

# Fitting qExponential-Power-law
def my_fit_qexp_power_law_old(x_train, n_it=50):
    from scipy.optimize import Bounds
    from scipy.optimize import minimize

    # Negative Log Likelihood
    def negative_log_likelihood(params, x):
        print(params)
        q, alpha, loc, scale = params
        pdf_values = qexp_power_law_pdf(x, q, alpha, loc=loc, scale=scale)
        log_pdf_values = np.log(pdf_values)
        return -np.sum(log_pdf_values)

    initial_guess = [1.25, 2, 1, np.std(x_train)] # Initial guesses for q and sigma
    bounds = Bounds([0, 1e-6, -np.inf, 1e-6], [2.95, np.inf, np.inf, np.inf], keep_feasible=True)

    # First fit with "smart" initialization
    result = minimize(negative_log_likelihood, initial_guess,
                      args=(x_train), method='SLSQP', bounds=bounds)
    if result.success:
        best_res, best_neg_ll = result, result.fun
    else:
        best_res, best_neg_ll = None, np.inf
    
    # Other fit with random initializations
    for temp_seed in range(n_it):
        np.random.seed(temp_seed)
        initial_guess = np.random.uniform(0, 1, 4)
        initial_guess[0] *= 2.95
        initial_guess[1] += 1e-6
        initial_guess[2] = 2*initial_guess[1] - 1
        initial_guess[3] += 1e-6

        result = minimize(negative_log_likelihood, initial_guess,
                          args=(x_train), method='SLSQP', bounds=bounds)
        if result.success and (result.fun < best_neg_ll):
            best_res = result
            best_neg_ll = result.fun
    if not isinstance(best_res, type(None)):
        return {'q':best_res.x[0], 'sigma':best_res.x[1],
                'loc':best_res.x[2], 'scale':best_res.x[3], 'neg_ll':best_res.fun}
    else:
        raise ValueError("Optimization failed.")

# Fitting qExponential-Power-law
def my_fit_qexp_power_law(x_train, n_it=50):
    from scipy.optimize import Bounds
    from scipy.optimize import minimize

    # Negative Log Likelihood
    def negative_log_likelihood(params, x):
        print(params)
        q, alpha, loc, scale = params
        pdf_values = qexp_power_law_pdf(x, q, alpha, loc=loc, scale=scale)
        log_pdf_values = np.log(pdf_values)
        return -np.sum(log_pdf_values)

    initial_guess = [1.25, np.std(x_train), 1, np.std(x_train)]  # Initial guesses for q and sigma
    bounds = Bounds([0, 1e-6, -np.inf, 1e-6], [2.95, np.inf, np.inf, np.inf], keep_feasible=True)

    # First fit with "smart" initialization
    result = minimize(negative_log_likelihood, initial_guess,
                      args=(x_train), method='SLSQP', bounds=bounds)
    if result.success:
        best_res, best_neg_ll = result, result.fun
    else:
        best_res, best_neg_ll = None, np.inf
    
    # Other fit with random initializations
    for temp_seed in range(n_it):
        np.random.seed(temp_seed)
        initial_guess = np.random.uniform(0, 1, 4)
        initial_guess[0] *= 2.95
        initial_guess[1] += 1e-6
        initial_guess[2] = np.min(x_train)*initial_guess[1]
        initial_guess[3] += 1e-6

        result = minimize(negative_log_likelihood, initial_guess,
                          args=(x_train), method='SLSQP', bounds=bounds)
        if result.success and (result.fun < best_neg_ll):
            best_res = result
            best_neg_ll = result.fun
    if not isinstance(best_res, type(None)):
        return {'q':best_res.x[0], 'sigma':best_res.x[1],
                'loc':best_res.x[2], 'scale':best_res.x[3], 'neg_ll':best_res.fun}
    else:
        raise ValueError("Optimization failed.")


# Total Swap Volume
with open(f'{folder}/usdc_weth_005.pickle', 'rb') as f:
    df = pickle.load(f)
df.index = pd.to_datetime(df.index)
df = df[ (df.Event == 'Swap_X2Y') | (df.Event == 'Swap_Y2X') ]
df = df.resample('1T').last()

# To avoid duplicates
potential_duplicated = df[ df.duplicated() ].block_number.unique()
sure_duplicated = list()
for block in potential_duplicated:
    temp_df = df[ df.block_number==block ]
    if (temp_df.duplicated()).sum()*2 == len(temp_df):
        sure_duplicated.append(block)
df = df[~ (df['block_number'].isin(sure_duplicated) & df.duplicated())]

# Joint volumes
start_date = datetime(2022, 1, 1, 00, 00, 00)
end_date = datetime(2022, 1, 2, 00, 00, 00)
x_train = np.abs(df[(df.index >= start_date) &\
                    (df.index < end_date)].amount0.dropna().values.astype(np.float64))
x_train = x_train[ x_train > 1 ] # Remove too small values

# Count the number of outliers
lq = np.quantile(x_train, 0.25)
uq = np.quantile(x_train, 0.75)
iqr = 1.5 * (uq - lq)
print('There are', 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train), '% outliers in the data')
outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

fitted_values = my_fit_qexp_power_law(x_train, n_it=100)
fitted_q, fitted_alpha, fitted_loc, fitted_scale = fitted_values['q'], fitted_values['alpha'], fitted_values['loc'], fitted_values['scale']
print(f"Fitted q: {fitted_q}, Fitted alpha: {fitted_alpha}, Fitted loc: {fitted_loc}, Fitted scale: {fitted_scale}")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.histplot(x_train, ax=ax, stat='probability', label='Data')

ax.set_yscale('log')
num_bins = len(ax.patches) # Number of bins
temp = ax.patches
hist_values = np.histogram(x_train, bins=num_bins, density=True)[0] #Obtain the histogram values
hist_values /= np.sum(hist_values) # Normalize the values
pdf = qexp_power_law_pdf(
    np.linspace(np.min(x_train), np.max(x_train), 10000), fitted_q, fitted_alpha, fitted_loc, fitted_scale)
sns.lineplot(x=np.linspace(np.min(x_train), np.max(x_train), 10000),
             y=pdf*np.max(hist_values)/np.max(pdf),
            ax=ax, color=sns.color_palette()[1], label='Scaled qGaussian PDF')

ax.legend()
plt.show()

# Swap X2Y Volume

# Swap Y2X Volume

#%% Volatility Analysis - 5 days

# Compute the five days volatility
n_day_vol = 1
x_train = list()
for day in pd.date_range(datetime(2021, 7, 1, 00, 00, 00),
                         datetime(2022, 12, 1, 00, 00, 00),
                         freq=pd.Timedelta(days=1)):
    start_date = datetime(day.year, day.month, day.day, 00, 00, 00)
    end_day = day + pd.Timedelta(days=n_day_vol)
    end_date = datetime(end_day.year, end_day.month, end_day.day, 00, 00, 00)
    temp_vals = df[(df.index >= start_date) & (df.index < end_date)].price.dropna().values.astype(np.float64)
    temp_vals = 100 * np.log(temp_vals[1:] / temp_vals[:-1])
    temp_vals = temp_vals[ temp_vals != 0] # Remove the zeros

    mean_val = np.mean(temp_vals)
    temp_vals -= mean_val # Remove the mean from the data
    
    x_train.append( np.mean(temp_vals**2) )

x_train = 1/np.array(x_train)

# Count the number of outliers
lq = np.quantile(x_train, 0.25)
uq = np.quantile(x_train, 0.75)
iqr = 1.5 * (uq - lq)
print('There are', 100*((x_train<lq-iqr) | (x_train>uq+iqr)).sum()/len(x_train), '% outliers in the data')
outlier_series = np.where((x_train<lq-iqr) | (x_train>uq+iqr), 1, 0)

fitted_values = qlogNormal.fit_two_sided(x_train)
fitted_q, fitted_sigma, fitted_mu, fitted_scale = fitted_values['q'], fitted_values['sigma'], fitted_values['mu'], fitted_values['scale']
print(f"Fitted q: {fitted_q}, Fitted sigma: {fitted_sigma}, Fitted mu: {fitted_mu}, Fitted scale: {fitted_scale}")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.histplot(x_train, ax=ax, stat='probability', label='Data')

num_bins = len(ax.patches) # Number of bins
temp = ax.patches
hist_values = np.histogram(x_train, bins=num_bins, density=True)[0] #Obtain the histogram values
hist_values /= np.sum(hist_values) # Normalize the values
pdf = qlogNormal.pdf_two_sided(np.linspace(np.min(x_train), np.max(x_train), 10000), fitted_q, fitted_sigma, fitted_mu, fitted_scale)

sns.lineplot(x=np.linspace(np.min(x_train), np.max(x_train), 10000),
             y=pdf*np.max(hist_values)/np.max(pdf),
            ax=ax, color=sns.color_palette()[1], label='Scaled qlogNormal Two Sided PDF')

ax.legend()
plt.show()

# Now, comparison between different pools:
with open(f'{folder}/usdc_weth_03.pickle', 'rb') as f:
    df03 = pickle.load(f)
df03.index = pd.to_datetime(df03.index)
df03 = df03[ (df03.Event == 'Swap_X2Y') | (df03.Event == 'Swap_Y2X') ]
df03 = df03.resample('1T').last()

with open(f'{folder}/usdc_weth_1.pickle', 'rb') as f:
    df1 = pickle.load(f)
df1.index = pd.to_datetime(df1.index)
df1 = df1[ (df1.Event == 'Swap_X2Y') | (df1.Event == 'Swap_Y2X') ]
df1 = df1.resample('1T').last()

for df_val, fee_level in zip([df03, df1], [0.3, 1]):
    print('fee =', fee_level)
    # Compute the five days volatility
    n_day_vol = 5
    x_train = list()
    for day in pd.date_range(datetime(2021, 7, 1, 00, 00, 00),
                            datetime(2022, 12, 1, 00, 00, 00),
                            freq=pd.Timedelta(days=1)):
        start_date = datetime(day.year, day.month, day.day, 00, 00, 00)
        end_day = day + pd.Timedelta(days=n_day_vol)
        end_date = datetime(end_day.year, end_day.month, end_day.day, 00, 00, 00)
        temp_vals = df_val[(df_val.index >= start_date) & (df_val.index < end_date)].price.dropna().values.astype(np.float64)
        temp_vals = 100 * np.log(temp_vals[1:] / temp_vals[:-1])
        temp_vals = temp_vals[ temp_vals != 0] # Remove the zeros

        mean_val = np.mean(temp_vals)
        temp_vals -= mean_val # Remove the mean from the data
        
        x_train.append( np.mean(temp_vals**2) )

    x_train = 1/np.array(x_train)

    fitted_values = qlogNormal.fit_two_sided(x_train)
    fitted_q, fitted_sigma, fitted_mu, fitted_scale = fitted_values['q'], fitted_values['sigma'], fitted_values['mu'], fitted_values['scale']
    print(f"Fitted q: {fitted_q}, Fitted sigma: {fitted_sigma}, Fitted mu: {fitted_mu}, Fitted scale: {fitted_scale}")

#%% Good data

start_date = datetime(2022, 1, 1, 00, 00, 00)
end_date = datetime(2022, 1, 2, 00, 00, 00)
x_train = df[(df.index >= start_date) & (df.index < end_date)].price.dropna().values.astype(np.float64)
x_train = 100 * np.log(x_train[1:] / x_train[:-1])
x_train = x_train[ x_train != 0] # Remove the zeros

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
        (stocks_hf.index < datetime(2020, 6, 2, 00, 00, 00)) ]['JPM'].values.astype(np.float64)
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
