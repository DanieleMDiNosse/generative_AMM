import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
cimport numpy as cnp
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
def optimize_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

@boundscheck(False)
@wraparound(False)
def group_by_freq(df, str freq):
    """
    Groups the dataframe by a specified frequency.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to group.
    freq : str
        The frequency to group by (e.g., '1D', '1W', '1M').
    
    Returns
    -------
    cumulative_dfs : generator
        A generator that yields cumulative DataFrames up to each frequency group.
    """
    df.index = pd.to_datetime(df.index)
    df_resampled = df.resample(freq)
    cumulative_df = pd.DataFrame()
    for _, group in tqdm(df_resampled, desc=f'Processing each {freq}'):
        cumulative_df = pd.concat([cumulative_df, group])
        yield cumulative_df.copy()

@boundscheck(False)
@wraparound(False)
def liquidity_dist_per_prange(pair_df, float min_price, float max_price, float delta=100):
    '''Calculate the distribution of liquidity amounts across price ranges.
    
    Parameters
    ----------
    pair_df : DataFrame
        The DataFrame containing the pair data.
    min_price : float
        The minimum price to consider. It will be the lower bound of the price range.
    max_price : float
        The maximum price to consider. It will be the upper bound of the price range.
    delta : float
        The width of each price bin.
    
    Returns
    -------
    amounts_per_bin_df : DataFrame
        A DataFrame containing the total liquidity amounts for each price bin.'''
    cdef:
        Py_ssize_t i, j
        float lower, upper, amount, bin_start
        cnp.ndarray[cnp.float32_t, ndim=1] price_bins_limited = np.arange(min_price, max_price + delta, delta, dtype=np.float32)
        dict amounts_per_bin_limited = {price_bin: 0.0 for price_bin in price_bins_limited}
        cnp.ndarray[cnp.float32_t, ndim=1] overlapping_bins

    filtered_data = pair_df.dropna(subset=['price_lower', 'price_upper', 'amount'])
    start, end = pair_df.index[0], pair_df.index[-1]
    filtered_data_limited = filtered_data[(filtered_data['price_lower'] >= min_price) & (filtered_data['price_upper'] <= max_price)]

    data = filtered_data_limited.to_numpy()

    for i in range(data.shape[0]):
        lower = data[i, filtered_data_limited.columns.get_loc('price_lower')]
        upper = data[i, filtered_data_limited.columns.get_loc('price_upper')]
        amount = data[i, filtered_data_limited.columns.get_loc('amount')]
        event = data[i, filtered_data_limited.columns.get_loc('Event')]

        if event == 'Burn':
            amount = -amount
        
        overlapping_bins = price_bins_limited[(price_bins_limited >= lower) & (price_bins_limited < upper)]
        
        for j in range(overlapping_bins.shape[0]):
            bin_start = overlapping_bins[j]
            amounts_per_bin_limited[bin_start] += amount

    amounts_per_bin_limited_df = pd.DataFrame(list(amounts_per_bin_limited.items()), columns=['price_bin', 'total_amount'])

    return amounts_per_bin_limited_df, [start, end]
