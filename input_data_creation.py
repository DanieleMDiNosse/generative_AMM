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
    bin_width = 10
    freq = '6h'
    delta = 100

    # Collect max and min prices for the liquidity_dist_per_prange function
    min_price = usdc_weth_005['price'].min()
    max_price = usdc_weth_005['price'].max()

    if os.path.exists(f'data/liquidity_dists_{freq}.pickle'):
        try:
            liquidity_dists = pickle.load(open(f'data/liquidity_dists_{freq}.pickle', 'rb'))
        except EOFError as e:
            print('Error loading liquidity distributions. Probabilly the file is empty or corrupted.')
            exit()
    else:
        # Initialize an empty list to store the results
        liquidity_dists = []
        
        # Process data in chunks and save intermediate results
        for i, df_chunk in enumerate(group_by_freq(usdc_weth_005, freq=freq)):
            liquidity_dist = liquidity_dist_per_prange(df_chunk, min_price=min_price, max_price=max_price, bin_width=bin_width)
            liquidity_dists.append(liquidity_dist)
            pickle.dump(liquidity_dists, open(f'data/liquidity_dists_{freq}.pickle', 'wb'))

        # Final save of all results
        pickle.dump(liquidity_dists, open(f'data/liquidity_dists_{freq}.pickle', 'wb'))
    
    price = usdc_weth_005[usdc_weth_005['Event'].isin(['Swap_Y2X', 'Swap_X2Y'])]['price']
    # sliced_price is a list of dataframes, each one containing the price data sliced by the liquidity distribution and the min and max price of the distribution
    sliced_prices = slice_price_by_liquidity_dists(liquidity_dists, price, freq=freq)
    liquidity_dfs = [v[0] for v in liquidity_dists]
    plot_price_and_liquidity(sliced_prices, liquidity_dfs, bin_width=bin_width, delta=delta, num_plot=5)
