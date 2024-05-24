'''This script contains some functions that can be used to visualize and analyze the data from the Uniswap V3 dataset.'''

import pickle
import os
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
from tqdm import tqdm
import imageio
import argparse
from eth_defi.uniswap_v3.liquidity import create_tick_delta_csv
from eth_defi.uniswap_v3.liquidity import create_tick_csv

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

def liquidity_dist_per_prange(pair_df, delta):
    liquidity_events = pair_df[pair_df['Event'].isin(['Mint', 'Burn'])]
    # liquidity_events = liquidity_events.drop(columns=['token1_address', 'transaction_price', 'tick', 'liquidity'])
    # Create equally spaced bins for the price ranges
    min_price = pair_df['price'].min()
    max_price = pair_df['price'].max()
    bins = np.arange(min_price, max_price, delta)
    # Create bins for the price ranges
    # pd.cut is used to segment and sort data values into bins
    # in this case we segment liquidity_events['price_lower'] into bins defined by the bins array
    price_bins = pd.cut(liquidity_events['price_lower'], bins=bins)

    # Multiply by -1 all the burn events to make them negative
    burns = np.where(liquidity_events['Event'] == 'Burn')
    liquidity_events['amount'].iloc[burns] = -1 * liquidity_events['amount'].iloc[burns]
    # Sum the liquidity amount for each bin
    liquidity_distribution = liquidity_events['amount'].groupby(price_bins, observed=False).sum()
    return liquidity_distribution

def liquidity_dist_per_tickid(mint_df_path, burn_df_path, pool_address):
    tick_delta_csv = create_tick_delta_csv(mint_df_path, burn_df_path)
    tick_csv = create_tick_csv(tick_delta_csv)
    tick_df = pd.read_csv(tick_csv)
    df = tick_df[tick_df.pool_contract_address == pool_address].sort_values(by="tick_id")
    df["liquidity_gross_delta"].astype(float).plot()
    return None

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
    usdc_weth = pickle.load(open('data/usdc_weth_05.pickle', 'rb'))
    usdc_weth = usdc_weth.drop_duplicates()

    if args.time_diffs:
        fig, ax = plt.subplots(2, 2, figsize=(13, 8), tight_layout=True)
        for i, event in enumerate(['Mint', 'Burn', 'Swap_X2Y', 'Swap_Y2X']):
            time_diff = time_diff_dist(usdc_weth, event)
            # plot histograms
            ax[i//2, i%2].hist(time_diff, bins=100, log=True)
            ax[i//2, i%2].set_title(f'Time differences for {event}')
            ax[i//2, i%2].set_yscale('log')
        plt.savefig('images/time_diffs.png')
    
    if args.liquidity_amounts:
        fig, ax = plt.subplots(2, 2, figsize=(13, 8), tight_layout=True)
        for i, event in enumerate(['Mint', 'Burn', 'Swap_X2Y', 'Swap_Y2X']):
            liquidity_amounts = liquidity_events_amount_dist(usdc_weth, event, token=1)
            # plot histograms
            ax[i//2, i%2].hist(liquidity_amounts, bins=100, log=True)
            ax[i//2, i%2].set_title(f'Liquidity amounts for {event}')
            ax[i//2, i%2].set_yscale('log')
        plt.savefig('images/liquidity_amounts.png')

    if args.liquidity_dist_price:
        liquidity_dist = liquidity_dist_per_prange(usdc_weth, delta=100)
        # Plot the distribution of liquidity across price ranges
        plt.figure(figsize=(10, 4))
        liquidity_dist.plot(kind='bar', edgecolor='k', alpha=0.7)
        plt.xlabel('Price Ranges')
        plt.ylabel('Total Liquidity')
        plt.title('Distribution of Liquidity for USDC/WETH pool')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('images/liquidity_dist_price.png')
        plt.show()

    if args.liquidity_dist_tickid:
        uni_eth_03 = '0x1d42064fc4beb5f8aaf85f4617ae8b3b5b8bd801'
        usdc_weth_05 = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640" #USDC-WETH 05
        usdc_weth_03 = "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8" #USDC-WETH 03
        dai_weth_ = "0x7bea39867e4169dbe237d55c8242a8f2fcdcc387" #USDC-WETH 1
        dai_usdc_ = "0x3416cf6c708da44db2624d63ea0aaef7113527c6" #DAI-USDC 001
        wbtc_usdc_03 = "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35" #WBTC-USDC 03
        wbtc_usdt_03 = "0x9db9e0e53058c89e5b94e29621a205198648425b" #WBTC-USDT 03
        liquidity_dist_per_tickid('data/uniswap-v3-mint.csv', 'data/uniswap-v3-burn.csv', usdc_weth_05)
        plt.savefig('images/liquidity_dist_tickid.png')
        plt.show()
    
    if args.sliding_window_difference:
        timestamps, differences = sliding_window_difference(usdc_weth, window_size=3000, step_size=50)
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

