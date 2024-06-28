'''This script contains some functions that can be used to visualize and analyze the data from the Uniswap V3 dataset.'''

import pickle
import random
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
from tqdm import tqdm
import imageio.v2 as imageio
import argparse
from datetime import datetime, timedelta
from eth_defi.uniswap_v3.liquidity import create_tick_delta_csv
from eth_defi.uniswap_v3.liquidity import create_tick_csv
from eth_defi.uniswap_v3.liquidity import get_pool_state_at_block
import os

def time_diff_dist(pair_df, event):
    df = pair_df[pair_df['Event'] == event]
    df.index = pd.to_datetime(df.index)
    unix_time = df.index.astype(int) // 10**9
    time_diffs = unix_time.diff()
    time_diffs = time_diffs[time_diffs != 0]
    return time_diffs

def liquidity_events_amount_dist(pair_df, event, token=1):
    df = pair_df[pair_df['Event'] == event]
    amount = df[f'amount{token}'].values
    return amount

def plot_price_and_liquidity(sliced_prices, liquidity_dists, bin_width, delta, num_plot):
    """
    Plots each pair of sliced_prices and liquidity_dists in a subplot with 1 row and 2 columns.

    Parameters:
    sliced_prices (list of pd.Series): List of sliced price Series.
    liquidity_dists (list of pd.DataFrame): List of DataFrames with timestamps as index.
    """

    print(sliced_prices[1000:1010])
    sliced_prices = sliced_prices[1000:1000+num_plot]
    num_pairs = len(sliced_prices)
    print(f'num_pairs: {num_pairs}')
    
    for i in range(num_pairs):
        price_slice = sliced_prices[i]['PriceSlice']
        min = sliced_prices[i]['Min'].values[0]
        max = sliced_prices[i]['Max'].values[0]
        p_t = price_slice.values[-1]
        liquidity_dist = liquidity_dists[i]

        _, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot sliced price
        axs[0].plot(price_slice.index, price_slice.values)
        axs[0].set_title('Price Slice')
        axs[0].set_xlabel('Timestamp')
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)
        axs[0].set_ylabel('Price')

        # Plot liquidity distribution
        bars = axs[1].bar(liquidity_dist['price_bin'], liquidity_dist['total_amount'], width=bin_width, edgecolor='black', color='red')
        for bar, price in zip(bars, liquidity_dist['price_bin']):
            if (p_t - delta) <= price <= (p_t + delta):
                bar.set_alpha(1)
            else:
                bar.set_alpha(0.3)

        axs[1].set_title('Liquidity Distribution')
        axs[1].set_xlabel('Price')
        axs[1].set_ylabel('Liquidity')
        
        # Add vertical lines for min and max
        axs[1].axvline(x=p_t, color='blue', linestyle='--', label='Min', alpha=0.4)
        
        # Adjust layout
        plt.tight_layout()
    plt.show()

def slice_price_by_liquidity_dists(liquidity_dists, price, freq):
    """
    For each DataFrame in liquidity_dists, slices the price series to include only the timestamps
    that fall within the first and last timestamp of the respective DataFrame.

    Parameters
    ----------
    liquidity_dists (list of [dataframe, (start, end)]): 
        List of [dataframe, (start, end)].
    price (pd.Series): 
        Series with timestamps as index.
    freq (str):
        Frequency of the liquidity dist ('1D', '6h', '1min', '30s').

    Returns
    -------
    list of pd.DataFrame: 
        List of DataFrames with sliced price Series for each DataFrame in liquidity_dists,
        including min and max prices in the slice.
    """
    sliced_prices = []

    for l in tqdm(liquidity_dists, desc='Slicing price'):
        # Get the end timestamp of the current liquidity distribution
        end_time_dist = l[1][1]

        # Determine start_time and end_time based on the freq
        if freq == '1D':
            start_time = end_time_dist.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(days=1) - timedelta(seconds=1)
        elif freq == '6h':
            start_time = end_time_dist.replace(minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(hours=6) - timedelta(seconds=1)
        elif freq == '1min':
            start_time = end_time_dist.replace(second=0, microsecond=0)
            end_time = start_time + timedelta(minutes=1) - timedelta(microseconds=1)
        elif freq == '30s':
            start_time = end_time_dist.replace(second=(end_time_dist.second // 30) * 30, microsecond=0)
            end_time = start_time + timedelta(seconds=30) - timedelta(microseconds=1)
        else:
            raise ValueError("Unsupported frequency. Please use '1D', '1min', or '30s'.")
        # print(f'range price: {start_time} - {end_time}')
        
        # Slice the price series to include only the relevant timestamps
        price_slice = price[(price.index >= start_time) & (price.index <= end_time)]
        max = np.max(price_slice)
        min = np.min(price_slice)
        price_slice_df = pd.DataFrame({'PriceSlice': price_slice, 'Min': min, 'Max': max})
        price_slice_df.index = price_slice.index
        sliced_prices.append(price_slice_df)

    return sliced_prices

def liquidity_dist_per_tickid(mint_df_path, burn_df_path, pool_address, tick_bin_size, compare_w_thegpraph=True):
    tick_delta_csv = create_tick_delta_csv(mint_df_path, burn_df_path)
    tick_csv = create_tick_csv(tick_delta_csv)
    tick_df = pd.read_csv(tick_csv)
    tick_df.to_csv("data/tick_df.csv")
    exit()
    df = tick_df[tick_df.pool_contract_address == pool_address].sort_values(by="tick_id")
    # Convert all the columns to float
    df = df.astype({col: 'float64' for col in df.columns if col != 'pool_contract_address'})
    # df.to_pickle("data/liquidity_dist_per_tickid.pkl")

    negative_gross_values = df[df.liquidity_gross_delta < 0]
    if len(negative_gross_values) > 0:
        print(f'Negative gross values: {len(negative_gross_values)} over {len(df)}. Discarding them...')
        df = df[df.liquidity_gross_delta >= 0]
    
    if tick_bin_size > 1:
        # Create a grouping column
        df['group'] = df.index // tick_bin_size
        # Group by the grouping column and apply the necessary aggregations
        result_df = df.groupby('group').agg(
            {'liquidity_gross_delta': 'sum', 'tick_id': 'last'}
        ).reset_index(drop=True)


    if compare_w_thegpraph:
        tickdelta_df = pd.read_csv(tick_delta_csv)
        print(tickdelta_df[tickdelta_df.pool_contract_address == pool_address].tail())
        compare_liq_per_tickid_w_thegraph(tickdelta_df, df, pool_address)

    return df

def compare_liq_per_tickid_w_thegraph(tick_delta_csv, df, pool_address):
    last_processed_block = tick_delta_csv[tick_delta_csv.pool_contract_address == pool_address].tail(1).block_number
    last_processed_block = int(last_processed_block.values[0])
    pool_state = get_pool_state_at_block(pool_address, last_processed_block)
    ticks = pool_state["ticks"]
    # get some random ticks from subgraph
    for i in range(10):
        random_tick = random.choice(ticks)

        # get the same tick from dataframe
        random_tick_df = df[df.tick_id == int(random_tick["tickIdx"])]

        # compare data
        assert int(random_tick_df.liquidity_gross_delta.values[0]) == int(random_tick["liquidityGross"])
        assert int(random_tick_df.liquidity_net_delta.values[0]) == int(random_tick["liquidityNet"])


def sliding_window_difference(liquidity_events_df, window_size, step_size):
    differences = []
    timestamps = []
    for start in tqdm(range(0, len(liquidity_events_df) - window_size, step_size)):
        end = start + window_size
        window = liquidity_events_df[start:end]
        mint_sum = window[window['Event'] == 'Mint'].sum()
        burn_sum = window[window['Event'] == 'Burn'].sum()
        
        diff_amount1 = mint_sum['amount1'] - burn_sum['amount1']
        diff_amount0 = mint_sum['amount0'] - burn_sum['amount0']
        
        differences.append((diff_amount1, diff_amount0))
        timestamps.append(window.index[int(window_size/2)])
    
    return np.array(timestamps), np.array(differences)
 
if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_diffs', action='store_true', help='Plot the time differences between events')
    parser.add_argument('--liquidity_amounts', action='store_true', help='Plot the liquidity amounts for events')
    parser.add_argument('--liquidity_dist_price', action='store_true', help='Plot the distribution of liquidity across price ranges')
    parser.add_argument('--liquidity_dist_tickid', action='store_true', help='Plot the distribution of liquidity across tickid')
    parser.add_argument('--sliding_window_difference', action='store_true', help='Plot the sliding window difference of liquidity')
    parser.add_argument('--all', action='store_true', help='Enable all plot options')
    args = parser.parse_args()

    if args.all:
        args.time_diffs = True
        args.liquidity_amounts = True
        args.liquidity_dist_price = True
        args.liquidity_dist_tickid = True
        args.sliding_window_difference = True

    # Load the data
    usdc_weth_005 = pickle.load(open('data/usdc_weth_005.pickle', 'rb'))
    usdc_weth_03 = pickle.load(open('data/usdc_weth_03.pickle', 'rb'))
    usdc_weth_1 = pickle.load(open('data/usdc_weth_1.pickle', 'rb'))
    pair_df = usdc_weth_03.drop_duplicates()

    if args.time_diffs:
        fig, ax = plt.subplots(2, 2, figsize=(13, 8), tight_layout=True)
        for i, event in enumerate(['Mint', 'Burn', 'Swap_X2Y', 'Swap_Y2X']):
            time_diff = time_diff_dist(pair_df, event)
            # plot histograms
            ax[i//2, i%2].hist(time_diff, bins=100, log=True)
            ax[i//2, i%2].set_title(f'Time differences for {event}')
            ax[i//2, i%2].set_yscale('log')
        plt.savefig('images/time_diffs.png')
    
    if args.liquidity_amounts:
        fig, ax = plt.subplots(2, 2, figsize=(13, 8), tight_layout=True)
        for i, event in enumerate(['Mint', 'Burn', 'Swap_X2Y', 'Swap_Y2X']):
            liquidity_amounts = liquidity_events_amount_dist(pair_df, event, token=1)
            # plot histograms
            ax[i//2, i%2].hist(liquidity_amounts, bins=100, log=True)
            ax[i//2, i%2].set_title(f'Liquidity amounts for {event}')
            ax[i//2, i%2].set_yscale('log')
        plt.savefig('images/liquidity_amounts.png')

    if args.liquidity_dist_price:
        min_price = pair_df['price'].min()
        max_price = pair_df['price'].max()
        bin_width = 50

        cumulative_dfs = group_by_freq(pair_df, freq='7D')
        # plot each distribution and create a video with all the plots
        images = []
        for i, df in tqdm(enumerate(cumulative_dfs)):
            liquidity_dist = liquidity_dist_per_prange(df, min_price, max_price, bin_width=bin_width)
            plt.figure(figsize=(12, 6))
            plt.bar(liquidity_dist['price_bin'], liquidity_dist['total_amount'], width=bin_width, edgecolor='black')
            plt.xlabel('Price Bin')
            plt.ylabel('Total Amount')
            plt.title('Adjusted Empirical Distribution of Amounts per Price Bin (Limited Range)')
            plt.xticks(liquidity_dist['price_bin'])
            plt.grid(axis='y')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'images/tmp/liquidity_dist_price_{i}.png')
            images.append(imageio.imread(f'images/tmp/liquidity_dist_price_{i}.png'))
            plt.close()
        # Get the list of image files in the images directory
        image_files = os.listdir('images/tmp')
        sorted_image_files = sorted(image_files)
        imageio.mimsave('images/liquidity_dist_price.gif', images, duration=0.5)

    if args.liquidity_dist_tickid:
        uni_eth_03 = '0x1d42064fc4beb5f8aaf85f4617ae8b3b5b8bd801'  # UNI/ETH 0.3%
        usdc_weth_005 = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" #USDC-WETH 005
        usdc_weth_03 = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8" #USDC-WETH 03
        dai_weth_ = "0x7bea39867e4169dbe237d55c8242a8f2fcdcc387" #USDC-WETH 1
        dai_usdc_ = "0x3416cf6c708da44db2624d63ea0aaef7113527c6" #DAI-USDC 001
        wbtc_usdc_03 = "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35" #WBTC-USDC 03
        wbtc_usdt_03 = "0x9db9e0e53058c89e5b94e29621a205198648425b" #WBTC-USDT 03
        df = liquidity_dist_per_tickid('data/uniswap-v3-mint.csv', 'data/uniswap-v3-burn.csv', usdc_weth_005, tick_bin_size=1, compare_w_thegpraph=False)
        df.to_csv('data/liquidity_dist_per_tickid.csv')
        plt.figure(figsize=(10, 4), tight_layout=True)
        plt.plot(df['tick_id'], df['liquidity_gross_delta'])
        plt.savefig('images/liquidity_dist_tickid.png')
        plt.show()
    
    if args.sliding_window_difference:
        timestamps, differences = sliding_window_difference(pair_df, window_size=3000, step_size=50)
        plt.figure(figsize=(10, 4))
        # plt.plot(timestamps, differences[:, 0], label='Amount1')
        # plt.plot(timestamps, differences[:, 1], label='Amount0')
        plt.plot(differences[:, 0], label='Amount1')
        plt.plot(differences[:, 1], label='Amount0')
        plt.xlabel('Timestamp')
        plt.ylabel('Liquidity Difference')
        plt.title('Liquidity Difference for USDC/WETH pool')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('images/sliding_window_difference.png')
        plt.show()

