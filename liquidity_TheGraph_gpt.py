import requests
import matplotlib.pyplot as plt
import json
import pandas as pd
from datetime import datetime

# Set up the new Uniswap v3 Subgraph Endpoint with your API key
API_KEY = 'dc2b2e1fd65c98965ab6c08bb8566316'  # Replace with your actual API key
SUBGRAPH_ID = '5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV'
url = f'https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}'

# Set headers
headers = {
    'Authorization': f'Bearer {API_KEY}'
}

# Token addresses for USDC and WETH (in lowercase)
USDC_ADDRESS = '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'
WETH_ADDRESS = '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2'

# Function to get the Pool ID
def get_pool_id():
    query_pool = '''
    {
      pools(
        where: {
          token0_in: ["USDC_ADDRESS", "WETH_ADDRESS"],
          token1_in: ["USDC_ADDRESS", "WETH_ADDRESS"],
          feeTier: 500
        }
      ) {
        id
        token0 {
          symbol
        }
        token1 {
          symbol
        }
        feeTier
      }
    }
    '''
    query_pool = query_pool.replace("USDC_ADDRESS", USDC_ADDRESS)
    query_pool = query_pool.replace("WETH_ADDRESS", WETH_ADDRESS)
    response = requests.post(url, json={'query': query_pool}, headers=headers)
    data = response.json()

    # Debugging: Print the raw response to inspect the data
    print("Raw Response:", data)

    # Check for errors in the response
    if 'errors' in data:
        raise Exception(f"GraphQL Error: {data['errors']}")

    pools = data['data']['pools']
    if not pools:
        raise Exception('Pool not found')
    
    pool_id = pools[0]['id']
    print(f'Pool ID: {pool_id}')
    return pool_id


# Function to fetch ticks for the pool
def fetch_ticks(pool_id):
    ticks = []
    skip = 0
    first = 1000  # Max items per request
    while True:
        query_ticks = '''
        {
          ticks(
            first: FIRST,
            skip: SKIP,
            where: { 
              poolAddress: "POOL_ID",
            },
            orderBy: tickIdx
          ) {
            tickIdx
            liquidityNet
            liquidityGross
            createdAtBlockNumber
            createdAtTimestamp
          }
        }
        '''
        query_ticks = query_ticks.replace("FIRST", str(first))
        query_ticks = query_ticks.replace("SKIP", str(skip))
        query_ticks = query_ticks.replace("POOL_ID", pool_id)
        # query_ticks = query_ticks.replace("START_BLOCK", str(start_block))
        # query_ticks = query_ticks.replace("END_BLOCK", str(end_block))
        
        response = requests.post(url, json={'query': query_ticks}, headers=headers)
        data = response.json()

        # Debugging: Check if there are any errors in the response
        if 'errors' in data:
            print("GraphQL Error:", data['errors'])
            raise Exception(f"GraphQL Error: {data['errors']}")

        # Debugging: Print the full response to inspect data
        print("Full Response:", data)

        # Check if ticks data exists in the response
        ticks_data = data.get('data', {}).get('ticks')
        if not ticks_data:
            break
        ticks.extend(ticks_data)
        skip += first
    return ticks


def get_block_by_human_timestamp(timestamp_str, closest='before', api_key='977JMGJYEWPTXN9D4NE3EMNGZJ5AXGQKU8'):
    """
    Get the block number closest to a specified human-readable timestamp from Etherscan.

    Parameters:
    - timestamp_str (str): The timestamp in 'YYYY-MM-DD:HH:MM:SS' format.
    - closest (str): Either 'before' or 'after' to get the block closest to the timestamp.
    - api_key (str): Your Etherscan API key.

    Returns:
    - int: The block number closest to the specified timestamp.
    """
    # Convert human-readable timestamp to UNIX timestamp
    timestamp = int(datetime.strptime(timestamp_str, '%Y-%m-%d:%H:%M:%S').timestamp())

    url = 'https://api.etherscan.io/api'
    params = {
        'module': 'block',
        'action': 'getblocknobytime',
        'timestamp': timestamp,
        'closest': closest,
        'apikey': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['status'] == '1':
        return int(data['result'])
    else:
        raise Exception(f"Error fetching block: {data['message']}")
    
def get_schema(url, api_key):
    headers = {'Authorization': f'Bearer {api_key}'}
    query = '''
    {
      __schema {
        types {
          name
          fields {
            name
            type {
              name
              kind
              ofType {
                name
                kind
              }
            }
          }
        }
      }
    }
    '''
    response = requests.post(url, json={'query': query}, headers=headers)
    schema = response.json()
    # Save to a file or print it for inspection
    with open("schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    # Print the schema (optional)
    print(json.dumps(schema, indent=2))
    return schema

# Function to create the bar plot
def plot_liquidity(data):
    tick_indexes = data['tickIdx']
    liquidity_gross = data['liquidityGross']

    plt.figure(figsize=(12, 6))
    plt.bar(tick_indexes, liquidity_gross, width=1)
    plt.xlabel('Tick Index')
    plt.ylabel('Gross Liquidity')
    plt.title('Gross Liquidity at Each Tick Index')
    plt.show()

# Main function
def main():
    # schema = get_schema(url, API_KEY)

    # timestamp_start = '2023-01-01:00:00:00'  # Start timestamp in 'YYYY-MM-DD:HH:MM:SS' format
    # timestamp_end = '2023-12-31:23:59:59'    # End timestamp in 'YYYY-MM-DD:HH:MM:SS' format
    # api_key = '977JMGJYEWPTXN9D4NE3EMNGZJ5AXGQKU8'

    # start_block = get_block_by_human_timestamp(timestamp_start, 'before', api_key)
    # end_block = get_block_by_human_timestamp(timestamp_end, 'after', api_key)

    # print(f"Start Block: {start_block}")
    # print(f"End Block: {end_block}")

    pool_id = get_pool_id()
    ticks = fetch_ticks(pool_id)
    # convert to a pandas dataframe
    data = pd.DataFrame(ticks)

    # Convert createdAtTimestamp values from UNIX timestamps to human-readable dates
    # data['createdAtTimestamp'] = pd.to_datetime(data['createdAtTimestamp'], unit='s')

    pd.to_pickle(data, 'data/liquidity_data.pkl')
    print(data.head())

    # plot_liquidity(data)
    

if __name__ == '__main__':
    main()
