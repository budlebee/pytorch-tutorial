# %%
import torch
from decouple import config
from binance import AsyncClient, BinanceSocketManager, Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd
import os
import matplotlib.pyplot as plt

binance_key = config('binance_key')
binance_secret = config('binance_secret')
client = Client(binance_key, binance_secret)
directory_of_python_script = os.path.dirname(os.path.abspath(__file__))

# %%
train_df = pd.read_csv(directory_of_python_script +
                       "/data/"+"BTC_kline_1minute_210101_211231.csv")

# %%
train_df_mean = train_df.mean()
train_df_std = train_df.std()

# z-score = ( value - mean )/std. about to 90~95% of the data is within 2 standard deviation of the mean
train_df_norm = (train_df - train_df_mean) / train_df_std
# %%
train_df_norm["volume"].plot()
train_df_norm["open_to_high_diff"].plot()

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

# %%
