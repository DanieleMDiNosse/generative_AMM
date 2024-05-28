
#%% Step 0: Import and base

import os
import pickle
import numpy as np
import pandas as pd
from eth_defi.provider.multi_provider import create_multi_provider_web3

json_rpc_url = 'INSERT YOUR API URL HERE, e.g. https://eth.llamarpc.com/sk_llama_00000000000000000000000000000000'
json_rpc_url = 'https://eth.llamarpc.com/sk_llama_252714c1e64c9873e3b21ff94d7f1a3f'
web3 = create_multi_provider_web3(json_rpc_url)

#%% Step 1: Interact with eth to get the raw data

from eth_defi.uniswap_v3.constants import UNISWAP_V3_FACTORY_CREATED_AT_BLOCK
from eth_defi.uniswap_v3.events import fetch_events_to_csv
from eth_defi.event_reader.json_state import JSONFileScanState

folder = '../data/tmp5'

# Take a snapshot of N blocks after Uniswap v3 deployment
start_block = UNISWAP_V3_FACTORY_CREATED_AT_BLOCK + 4_980_000
end_block = UNISWAP_V3_FACTORY_CREATED_AT_BLOCK + 10_000_000

# Stores the last block number of event data we store
state = JSONFileScanState(folder+"/uniswap-v3-price-scan.json")

# Load the events and write them into a CSV file.
# Several different CSV files are created,
# each for one event type: swap, pool created, mint, burn
web3 = fetch_events_to_csv(
    json_rpc_url,
    state,
    start_block=start_block,
    end_block=end_block,
    output_folder=folder,
    # Configure depending on what's eth_getLogs
    # limit of your JSON-RPC provider and also
    # how often you want to see progress bar updates
    max_blocks_once=222,
    # Do reading and decoding in parallel threads
    max_threads=8,
)

#%% Step 2: Select a specific pair and extract the data, ready for use

#------------------------------------------ Some general info...

folder = '../data/tmp3'
swap_df = pd.read_csv(f"{folder}/uniswap-v3-swap.csv")
print(f"We have total {len(swap_df):,} Uniswap swap events in the loaded dataset")
column_names = ", ".join([n for n in swap_df.columns])
print("Swap data columns are:", column_names)
print('\n')
swap_df.tail(10)

created_df = pd.read_csv(f"{folder}/uniswap-v3-poolcreated.csv")
print(f"We have total {len(created_df):,} created pools in the loaded dataset")
column_names = ", ".join([n for n in created_df.columns])
print("Created pools columns are:", column_names)
print('\n')
created_df.tail(10)

mint_df = pd.read_csv(f"{folder}/uniswap-v3-mint.csv")
print(f"We have total {len(mint_df):,} mint events in the loaded dataset")
column_names = ", ".join([n for n in mint_df.columns])
print("Mint data columns are:", column_names)
print('\n')
mint_df.tail(10)

burn_df = pd.read_csv(f"{folder}/uniswap-v3-burn.csv")
print(f"We have total {len(burn_df):,} burn events in the loaded dataset")
column_names = ", ".join([n for n in burn_df.columns])
print("Burn data columns are:", column_names)
print('\n')
burn_df.tail(10)

#------------------------------------------ Fetch USDC-WETH 005 details
from eth_defi.uniswap_v3.pool import fetch_pool_details
from eth_defi.uniswap_v3.price import get_onchain_price

pool_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" #USDC-WETH 005
pool_address = "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8" #USDC-WETH 03
pool_address = "0x7bea39867e4169dbe237d55c8242a8f2fcdcc387" #USDC-WETH 1
pool_address = "0x3416cf6c708da44db2624d63ea0aaef7113527c6" #DAI-USDC 001
pool_address = "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35" #WBTC-USDC 03
pool_address = "0x9db9e0e53058c89e5b94e29621a205198648425b" #WBTC-USDT 03
pool_details = fetch_pool_details(web3, pool_address)

reverse_quote_base = False

print(pool_details)
print("token0 is", pool_details.token0)
print("token1 is", pool_details.token1)
print('\n')

# Select only the specific pool
swap_df, created_df, mint_df, burn_df = [
    df.loc[df["pool_contract_address"] == pool_address.lower()] for df in [
        swap_df, created_df, mint_df, burn_df]]

# Add the event type
created_df['Event'] = ['Creation']*len(created_df)
mint_df['Event'] = ['Mint']*len(mint_df)
burn_df['Event'] = ['Burn']*len(burn_df)

# Add price and value columns
def tick2price(row, tick_col):
    # USDC/WETH pool has reverse token order, so let's flip it WETH/USDC
    return float(pool_details.convert_price_to_human(row[tick_col], reverse_token_order=reverse_quote_base))
# def convert_value(row):
#     # USDC is token0 and amount0
#     return abs(float(row["amount0"])) / (10**pool_details.token0.decimals)
swap_df["price"] = swap_df.apply(tick2price, axis=1, args=("tick",))
mint_df["price_lower"] = mint_df.apply(tick2price, axis=1, args=("tick_upper",))
mint_df["price_upper"] = mint_df.apply(tick2price, axis=1, args=("tick_lower",))
burn_df["price_lower"] = burn_df.apply(tick2price, axis=1, args=("tick_upper",))
burn_df["price_upper"] = burn_df.apply(tick2price, axis=1, args=("tick_lower",))

# Normalize amounts
swap_df['amount0'] = swap_df.apply(
    lambda x: float(int(x['amount0']) / 10**pool_details.token0.decimals), axis=1)
swap_df['amount1'] = swap_df.apply(
    lambda x: float(int(x['amount1']) / 10**pool_details.token1.decimals), axis=1)
swap_df['Event'] = np.where(swap_df.amount0>0, 'Swap_X2Y', 'Swap_Y2X')
delta_decimals = pool_details.token0.decimals - pool_details.token1.decimals
swap_df['liquidity'] = swap_df.apply(
    lambda x: float(int(x['liquidity']) * 10**(delta_decimals)), axis=1)

swap_df["transaction_price"] = np.where(reverse_quote_base,
                                        np.abs(swap_df.amount0 / swap_df.amount1),
                                        np.abs(swap_df.amount1 / swap_df.amount0))

# Normalize mint and burn ammounts
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
df.drop(columns=['token0_symbol', 'token1_symbol', 'tx_hash', 'sqrt_price_x96',
                 'token0_address', 'token0_address', 'fee',
                 'factory_contract_address', 'pool_contract_address'], inplace=True)

df.index = df.timestamp; df.drop(columns=['timestamp'], inplace=True)
df = df.sort_index()
df

with open(f'{folder}/wbtc_usdc_03.pickle', 'wb') as f:
    pickle.dump(df, f)

''' OSSERVAZIONI E COMMENTI VARI:
SWAP: price è riferito al prezzo alla fine del blocco; liquidity è la liquidità virtuale;
amount0 e amount1 corrispondono al punto di vista della pool. Per esempio, amount0>0 =>
    LT sta comprando 1. Inoltre, da come ho capito, le fee sono già state sottratte.
Ino
price_transaction l'ho calcolato io come abs( amount0 / amount1 ).

MINT: amount di fatto corrisponde a L aggiunto o sottratto, dove L è la liquidità virtuale
'''

#%% #%% Step 3: Merge the data to obtain the ultimate dataset

#------------------------------------------ Some general info...

folder = '../data/tmp5'
first_block = 0
first_block = 17369621

swap_df = pd.read_csv(f"{folder}/uniswap-v3-swap.csv")
swap_df = swap_df[ swap_df.block_number > first_block ]
print(f"We have total {len(swap_df):,} Uniswap swap events in the loaded dataset")
column_names = ", ".join([n for n in swap_df.columns])
print("Swap data columns are:", column_names)
print('\n')

created_df = pd.read_csv(f"{folder}/uniswap-v3-poolcreated.csv")
created_df = created_df[ created_df.block_number > first_block ]
print(f"We have total {len(created_df):,} created pools in the loaded dataset")
column_names = ", ".join([n for n in created_df.columns])
print("Created pools columns are:", column_names)
print('\n')

mint_df = pd.read_csv(f"{folder}/uniswap-v3-mint.csv")
mint_df = mint_df[ mint_df.block_number > first_block ]
print(f"We have total {len(mint_df):,} mint events in the loaded dataset")
column_names = ", ".join([n for n in mint_df.columns])
print("Mint data columns are:", column_names)
print('\n')

burn_df = pd.read_csv(f"{folder}/uniswap-v3-burn.csv")
burn_df = burn_df[ burn_df.block_number > first_block ]
print(f"We have total {len(burn_df):,} burn events in the loaded dataset")
column_names = ", ".join([n for n in burn_df.columns])
print("Burn data columns are:", column_names)
print('\n')

#------------------------------------------ Fetch USDC-WETH 005 details
from eth_defi.uniswap_v3.pool import fetch_pool_details
from eth_defi.uniswap_v3.price import get_onchain_price

pool_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640" #USDC-WETH 005
pool_address = "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8" #USDC-WETH 03
pool_address = "0x7bea39867e4169dbe237d55c8242a8f2fcdcc387" #USDC-WETH 1
pool_address = "0x3416cf6c708da44db2624d63ea0aaef7113527c6" #DAI-USDC 001
pool_address = "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35" #WBTC-USDC 03
pool_address = "0x9db9e0e53058c89e5b94e29621a205198648425b" #WBTC-USDT 03

pool_details = fetch_pool_details(web3, pool_address)
reverse_quote_base = True

print(pool_details)
print("token0 is", pool_details.token0)
print("token1 is", pool_details.token1)
print('\n')

# Select only the specific pool
swap_df, created_df, mint_df, burn_df = [
    df.loc[df["pool_contract_address"] == pool_address.lower()] for df in [
        swap_df, created_df, mint_df, burn_df]]

# Add the event type
created_df['Event'] = ['Creation']*len(created_df)
mint_df['Event'] = ['Mint']*len(mint_df)
burn_df['Event'] = ['Burn']*len(burn_df)

# Add price and value columns
def tick2price(row, tick_col):
    # USDC/WETH pool has reverse token order, so let's flip it WETH/USDC
    return float(pool_details.convert_price_to_human(row[tick_col], reverse_token_order=reverse_quote_base))
# def convert_value(row):
#     # USDC is token0 and amount0
#     return abs(float(row["amount0"])) / (10**pool_details.token0.decimals)
swap_df["price"] = swap_df.apply(tick2price, axis=1, args=("tick",))
mint_df["price_lower"] = mint_df.apply(tick2price, axis=1, args=("tick_upper",))
mint_df["price_upper"] = mint_df.apply(tick2price, axis=1, args=("tick_lower",))
burn_df["price_lower"] = burn_df.apply(tick2price, axis=1, args=("tick_upper",))
burn_df["price_upper"] = burn_df.apply(tick2price, axis=1, args=("tick_lower",))

# Normalize amounts
swap_df['amount0'] = swap_df.apply(
    lambda x: float(int(x['amount0']) / 10**pool_details.token0.decimals), axis=1)
swap_df['amount1'] = swap_df.apply(
    lambda x: float(int(x['amount1']) / 10**pool_details.token1.decimals), axis=1)
swap_df['Event'] = np.where(swap_df.amount0>0, 'Swap_X2Y', 'Swap_Y2X')
delta_decimals = pool_details.token0.decimals - pool_details.token1.decimals
swap_df['liquidity'] = swap_df.apply(
    lambda x: float(int(x['liquidity']) * 10**(delta_decimals)), axis=1)

swap_df["transaction_price"] = np.where(reverse_quote_base,
                                        np.abs(swap_df.amount0 / swap_df.amount1),
                                        np.abs(swap_df.amount1 / swap_df.amount0))

# Normalize mint and burn ammounts
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
df.drop(columns=['token0_symbol', 'token1_symbol', 'tx_hash', 'sqrt_price_x96',
                 'token0_address', 'token1_address', 'fee',
                 'factory_contract_address', 'pool_contract_address'], inplace=True)

df.index = df.timestamp; df.drop(columns=['timestamp'], inplace=True)
df = df.sort_index()

if first_block == 0:
    df_first = df.copy()
else:
    df = pd.concat([df_first, df])
    with open(f'../data/wbtc_usdc_005.pickle', 'wb') as f:
        pickle.dump(df, f)

#%% Liquidity problem

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

folder = '../data/tmp3'
with open(f'{folder}/usdc_weth_005.pickle', 'rb') as f:
    df = pickle.load(f)

potential_duplicated = df[ df.duplicated() ].block_number.unique()
sure_duplicated = list()
for block in potential_duplicated:
    temp_df = df[ df.block_number==block ]
    if (temp_df.duplicated()).sum()*2 == len(temp_df):
        sure_duplicated.append(block)
df = df[~ (df['block_number'].isin(sure_duplicated) & df.duplicated())]

df.index = pd.to_datetime(df.index)
df = df[['amount', 'price_lower', 'price_upper', 'Event']]

# In this step, I just need the swap data
df_mint = df[ df.Event == 'Mint' ]
df_burn = df[ df.Event == 'Burn' ]

for t_burn in tqdm(range(len(df_burn)), desc='Iterating over burn events'):
    # Initialize the amount to absorb
    to_absorb = df_burn['amount'].iloc[t_burn]
    t = 0
    #to_remove = list()

    while (to_absorb>0) and (df_mint.index[t] <= df_burn.index[t_burn]):
        if np.isclose(df_burn['price_lower'].iloc[t_burn], df_mint['price_lower'].iloc[t]) &\
            np.isclose(df_burn['price_upper'].iloc[t_burn], df_mint['price_upper'].iloc[t]):
            if np.isclose(to_absorb, df_mint['amount'].iloc[t]):
                df_mint['amount'].iloc[t] = 0
                to_absorb = 0
                #to_remove.append(t)
            elif to_absorb > df_mint['amount'].iloc[t]:
                to_absorb -= df_mint['amount'].iloc[t]
                df_mint['amount'].iloc[t] = 0
                #to_remove.append(t)
            else:
                df_mint['amount'].iloc[t] -= to_absorb
                to_absorb = 0
        t += 1
    
    # for t in to_remove[::-1]:
    #     df_mint.drop(df_mint.index[t], inplace=True)

    if not np.isclose(to_absorb, 0):
        print('Errore!!!')
        print(t_burn)

df_mint_temp = df_mint[ df_mint.index<=pd.to_datetime('2021-05-20 09:14:48') ]

df_mint_temp[ (df_mint_temp.price_upper > 10_000) & (df_mint_temp.price_upper < 3_500) & (df_mint_temp.price_lower > 3_450) & (df_mint_temp.price_lower < 2431)]



#%% VECCHIO SCHIFO

import os
# Set environment variable for this script and any subprocesses
os.environ["TRADING_STRATEGY_API_KEY"] = "secret-token:tradingstrategy-8aaeef270e19616ed4be32ec9d47f2823b6882ca3d957f9438c4c40e090a6c14"
# Example: running a command that uses the environment variable
os.system('echo $TRADING_STRATEGY_API_KEY')

import numpy as np
import pandas as pd

# Search for the pair with the given token symbols
def search_pair(token0_symbol, token1_symbol):
    # search for the pair with the given token symbols regardless the order
    pair_f = pairs[(pairs['token0_symbol'] == token0_symbol) & (pairs['token1_symbol'] == token1_symbol)]
    pair_r = pairs[(pairs['token0_symbol'] == token1_symbol) & (pairs['token1_symbol'] == token0_symbol)]
    # concatenate the two results
    pair = pd.concat([pair_f, pair_r])
    # sort the result by buy_volume_all_time + sell_volume_all_time
    pair = pair.sort_values(by=['buy_volume_all_time', 'sell_volume_all_time'], ascending=False)
    return pair

# Search for the pairs on the given exchange
def search_exchange(exchange):
    return pairs[pairs['dex_type'] == exchange]

# Search by pair id
def search_pair_by_id(pair_id):
    return pairs[pairs['pair_id'] == pair_id]

# Read the parquet file containing all the pairs
pairs = pd.read_parquet('../data/pair-universe')
print(f'Features of the pairs parquet dataframe:\n\t{pairs.columns}')

# Search for the pair with the given token symbols
pair = search_pair('WETH', 'USDC')
pair


from tradingstrategy.client import Client
from tradingstrategy.trade import Trade
from tradingstrategy.timebucket import TimeBucket
from datetime import datetime

# Create a client
client = Client.create_jupyter_client()


# Fetch the candles for the given pair
delta_t = TimeBucket.m1
start_time = datetime(2024, 5, 7, 00, 00, 00)  # 0:00:00 on January 1, 2024
end_time = datetime(2024, 5, 13, 23, 59, 00)    # 17:00:00 on January 1, 2024

candle_pairs = client.fetch_candles_by_pair_ids(
    [1], bucket=delta_t, start_time=start_time, end_time=end_time,
    max_bytes=None)
candle_pairs

columns2consider = ['price', 'volume', 'buys', 'sells', 'buy_volume', 'sell_volume']
candle_pairs['price'] = candle_pairs['close']
candle_pairs = candle_pairs[columns2consider]

to_save = pd.DataFrame(index=pd.date_range(start_time, end_time, freq='T'),
                       columns=columns2consider)
for idx in to_save.index:
    if idx in candle_pairs.index:
        to_save.loc[idx] = candle_pairs.loc[idx]
        prev_price = candle_pairs.loc[idx, 'price']
    else:
        to_save.loc[idx] = [0.]*len(columns2consider)
        to_save.loc[idx, 'price'] = prev_price


from tradingstrategy.liquidity import GroupedLiquidityUniverse

prova = client.fetch_liquidity_by_pair_ids(
    [1], bucket=delta_t, start_time=start_time, end_time=end_time,
    max_bytes=None)
prova


from tqdm.auto import tqdm
import pyarrow.parquet as pa

table = pa.read_table('../data/liquidity-samples-1m.parquet') 
df = table.to_pandas()


