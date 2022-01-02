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


def get_2yr_data(sym, itv):
    # take data from binance
    data_2021 = client.futures_historical_klines(
        symbol=sym+"USDT",
        interval=itv,
        start_str='2021-01-01',
        end_str="2021-12-31"
    )
    data_2020 = client.futures_historical_klines(
        symbol=sym+"USDT",
        interval=itv,
        start_str='2020-01-01',
        end_str="2020-12-31"
    )
    df_2021 = pd.DataFrame(data_2021, columns=['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time',
                                               'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df_2021.to_csv(directory_of_python_script +
                   "/data/"+sym+"_kline_"+itv+"_210101_211231.csv", sep=',')
    df_2020 = pd.DataFrame(data_2020, columns=['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time',
                                               'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df_2020.to_csv(directory_of_python_script +
                   "/data/"+sym+"_kline_"+itv+"_200101_201231.csv", sep=',')


get_2yr_data("SHIB", "1h")

# train_df = pd.read_csv(directory_of_python_script +
#                       "/data/"+"kline_1h_210101_211231.csv")
#plt.plot([1, 2, 3, 4])
# plt.show()
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
