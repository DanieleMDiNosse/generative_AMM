from eth_defi.provider.multi_provider import create_multi_provider_web3
import os
import pandas as pd
import numpy as np
import pickle
from web3 import Web3, HTTPProvider
from eth_defi.uniswap_v3.constants import UNISWAP_V3_FACTORY_CREATED_AT_BLOCK
from eth_defi.uniswap_v3.events import fetch_events_to_csv
from eth_defi.event_reader.json_state import JSONFileScanState
import pandas as pd
from eth_defi.uniswap_v3.pool import fetch_pool_details

def tick2price(row, tick_col):
    # If reverse_quote_order is True, the price is token1/token0
    return float(pool_details.convert_price_to_human(row[tick_col], reverse_token_order=reverse_quote_base))

if __name__ == '__main__':

    # Get your node JSON-RPC URL
    # interactively when you run the notebook.
    # If you are running from command line you can also pass this as JSON_RPC_ETHEREUM environment variable
    # 'https://eth.llamarpc.com/sk_llama_252714c1e64c9873e3b21ff94d7f1a3f'
    # 'https://mainnet.infura.io/v3/5f38fb376e0548c8a828112252a6a588'
    # json_rpc_url = os.environ.get("JSON_RPC_ETHEREUM")
    json_rpc_url = "https://eth.llamarpc.com/sk_llama_252714c1e64c9873e3b21ff94d7f1a3f https://mainnet.infura.io/v3/5f38fb376e0548c8a828112252a6a588 https://eth-mainnet.g.alchemy.com/v2/eq9r2pPrnkczHi1MuJmXPMAc3nn3kc4F"
    web3 = create_multi_provider_web3(json_rpc_url)

    # Take a snapshot of end_block-start_block blocks after Uniswap v3 deployment
    start_block = UNISWAP_V3_FACTORY_CREATED_AT_BLOCK + 4_980_000
    end_block = UNISWAP_V3_FACTORY_CREATED_AT_BLOCK + 10_000_000
    folder = "/home/danielemdn/Documents/repositories/generative_AMM/data/tmp"
    reverse_quote_base = False
    
    # Stores the last block number of event data we store
    state = JSONFileScanState("data/tmp/uniswap-v3-price-scan.json")

    print(f"Data snapshot range set to {start_block:,} - {end_block:,}")

    # Load the events and write them into a CSV file.
    # Several different CSV files are created,
    # each for one event type: swap, pool created, mint, burn
    web3 = fetch_events_to_csv(
        json_rpc_url,
        state,
        start_block=start_block,
        end_block=end_block,
        output_folder=f"{folder}",
        # Configure depending on what's eth_getLogs
        # limit of your JSON-RPC provider and also
        # how often you want to see progress bar updates
        max_blocks_once=500,
        # Do reading and decoding in parallel threads
        max_threads=4,
    )

    # Load the CSV files into Pandas DataFrames
    created_df = pd.read_csv(f"{folder}/uniswap-v3-poolcreated.csv")
    swap_df = pd.read_csv(f"{folder}/uniswap-v3-swap.csv")
    mint_df = pd.read_csv(f"{folder}/uniswap-v3-mint.csv")
    burn_df = pd.read_csv(f"{folder}/uniswap-v3-burn.csv")

    # Select the pool you want to analyze
    pool_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" #USDC-WETH 005
    pool_details = fetch_pool_details(web3, pool_address) # It returns a PoolDetails object
    pool_details_dict = {'token0': str(pool_details.token0), 'token1': str(pool_details.token1),
                         'decimals0': int(pool_details.token0.decimals), 'decimals1': int(pool_details.token1.decimals)}

    # Save the pool_details object for future reference
    with open(f'{folder}/pool_details.pickle', 'wb') as f:
        pickle.dump(pool_details_dict, f)

    print(pool_details)
    print("token0 is", pool_details.token0)
    print("token1 is", pool_details.token1)

    # Filter the DataFrames to only include the events of the pool you want to analyze
    swap_df, created_df, mint_df, burn_df = [
        df.loc[df["pool_contract_address"] == pool_address.lower()] for df in [
            swap_df, created_df, mint_df, burn_df]]

    # Add the event type
    created_df['Event'] = ['Creation']*len(created_df)
    mint_df['Event'] = ['Mint']*len(mint_df)
    burn_df['Event'] = ['Burn']*len(burn_df)

    swap_df["price"] = swap_df.apply(tick2price, axis=1, args=("tick",))
    mint_df["price_lower"] = mint_df.apply(tick2price, axis=1, args=("tick_upper",))
    mint_df["price_upper"] = mint_df.apply(tick2price, axis=1, args=("tick_lower",))
    burn_df["price_lower"] = burn_df.apply(tick2price, axis=1, args=("tick_upper",))
    burn_df["price_upper"] = burn_df.apply(tick2price, axis=1, args=("tick_lower",))

    # Normalize amounts based on token decimals
    swap_df['amount0'] = swap_df.apply(
        lambda x: float(int(x['amount0']) / 10**pool_details.token0.decimals), axis=1)
    swap_df['amount1'] = swap_df.apply(
        lambda x: float(int(x['amount1']) / 10**pool_details.token1.decimals), axis=1)
    swap_df['Event'] = np.where(swap_df.amount0>0, 'Swap_X2Y', 'Swap_Y2X')
    delta_decimals = pool_details.token0.decimals - pool_details.token1.decimals
    swap_df['liquidity'] = swap_df.apply(
        lambda x: float(int(x['liquidity']) * 10**(delta_decimals)), axis=1)

    # Calculate the transaction price for each swap
    swap_df["transaction_price"] = np.where(reverse_quote_base,
                                            np.abs(swap_df.amount0 / swap_df.amount1),
                                            np.abs(swap_df.amount1 / swap_df.amount0))

    # Normalize mint and burn ammounts based on token decimals
    mint_df['amount0'] = mint_df.apply(
        lambda x: float(int(x['amount0']) / 10**pool_details.token0.decimals), axis=1)
    mint_df['amount1'] = mint_df.apply(
        lambda x: float(int(x['amount1']) / 10**pool_details.token1.decimals), axis=1)
    mint_df['amount'] = mint_df.apply(
        lambda x: float(int(x['amount']) * 10**(delta_decimals)), axis=1)
    burn_df['amount0'] = burn_df.apply(
        lambda x: float(int(x['amount0']) / 10**pool_details.token0.decimals), axis=1)
    burn_df['amount1'] = burn_df.apply(
        lambda x: float(int(x['amount1']) / 10**pool_details.token1.decimals), axis=1)
    burn_df['amount'] = burn_df.apply(
        lambda x: float(int(x['amount']) * 10**(delta_decimals)), axis=1)

    # List of all DataFrames
    dfs = [df.loc[df["pool_contract_address"] == pool_address.lower()] for df in [
        swap_df, created_df, mint_df, burn_df]]

    # Create a union of all columns
    all_columns = list(set().union(*[df.columns for df in dfs]))

    # Reindex each DataFrame to ensure all columns are present, filling missing values with NaN
    dfs = [df.reindex(columns=all_columns) for df in dfs]

    # Concatenate DataFrames along rows
    df = pd.concat(dfs, ignore_index=True)

    # Remove some useless columns
    df.drop(columns=['token0_symbol', 'token1_symbol', 'sqrt_price_x96',
                    'token0_address', 'token0_address', 'fee',
                    'factory_contract_address', 'pool_contract_address'], inplace=True)

    df.index = df.timestamp; df.drop(columns=['timestamp'], inplace=True)
    df = df.sort_index()

    with open(f'{folder}/weth_usdc_03_{start_block}_{end_block}.pickle', 'wb') as f:
        pickle.dump(df, f)
