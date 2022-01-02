import torch
from decouple import config
from binance import AsyncClient, BinanceSocketManager, Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd
import os

binance_key = config('binance_key')
binance_secret = config('binance_secret')
client = Client(binance_key, binance_secret)

data = client.futures_historical_klines(
    symbol="BTCUSDT",
    interval='1h',
    start_str='2021-01-01',
    end_str="2021-12-31"
)
df = pd.DataFrame(data, columns=['', 'open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time',
                  'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
df.to_csv("/app/data_dir/BTC_kline_1h_210101_211231.csv", sep=',')

'''

directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(directory_of_python_script +
                       "/data/"+"kline_1minute_210101_211231.csv")
test_df = pd.read_csv(directory_of_python_script +
                      "/data/"+"kline_1minute_200101_201231.csv")
train_df_mean = train_df.mean()
train_df_std = train_df.std()

# z-score = ( value - mean )/std. about to 90~95% of the data is within 2 standard deviation of the mean
train_df_norm = (train_df - train_df_mean) / train_df_std

norm_open_price = train_df_norm['open_price']
norm_high_price = train_df_norm['high_price']
norm_low_price = train_df_norm['low_price']
norm_close_price = train_df_norm['close_price']
norm_volume = train_df_norm['volume']

print(train_df_norm.head())
'''
