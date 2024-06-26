'''This script creates the input data for the CDF estimation. The input data must be composed by the history of the marginal price and the liquidity distribution.'''
import pickle
import pandas as pd
import os
from utils_input_data import optimize_dataframe, group_by_freq, liquidity_dist_per_prange
from data_processing import plot_price_and_liquidity, slice_price_by_liquidity_dists

if __name__ == '__main__':
    usdc_weth_005 = pickle.load(open('data/usdc_weth_005.pickle', 'rb'))
    usdc_weth_005.index = pd.to_datetime(usdc_weth_005.index)
    usdc_weth_005 = optimize_dataframe(usdc_weth_005)

    # Hyperparameters
    delta = 50
    freq = '1D'
    nu = 50

    # Collect max and min prices for the liquidity_dist_per_prange function
    min_price = usdc_weth_005['price'].min()
    max_price = usdc_weth_005['price'].max()

    if os.path.exists(f'data/liquidity_dists_{freq}.pickle'):
        liquidity_dists = pickle.load(open(f'data/liquidity_dists_{freq}.pickle', 'rb'))
    else:
        # Initialize an empty list to store the results
        liquidity_dists = []
        
        # Process data in chunks and save intermediate results
        for i, df_chunk in enumerate(group_by_freq(usdc_weth_005, freq=freq)):
            liquidity_dist = liquidity_dist_per_prange(df_chunk, min_price=min_price, max_price=max_price, delta=delta)
            liquidity_dists.append(liquidity_dist)
            pickle.dump(liquidity_dists, open(f'data/liquidity_dists_{freq}.pickle', 'wb'))

        # Final save of all results
        pickle.dump(liquidity_dists, open(f'data/liquidity_dists_{freq}.pickle', 'wb'))
    
    price = usdc_weth_005[usdc_weth_005['Event'].isin(['Swap_Y2X', 'Swap_X2Y'])]['price']
    # sliced_price is a list pandas series, where each series is the price within the time interval covered by the corresponding liquidity distribution.
    sliced_prices = slice_price_by_liquidity_dists(liquidity_dists, price)
    liquidity_dfs = [v[0] for v in liquidity_dists]
    plot_price_and_liquidity(sliced_prices, liquidity_dfs, delta=delta, nu=nu, num_plot=5)
