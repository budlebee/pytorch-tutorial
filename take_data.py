# %%
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


def take_spot_data(symbol="BTC", interval="1m", start="2022-01-01", end="2022-01-31"):
    data = client.get_historical_klines(
        symbol=symbol+"USDT",
        interval=interval,
        start_str=start,
        end_str=end
    )
    df = pd.DataFrame(data, columns=['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time',
                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df.to_csv(directory_of_python_script + "/data/spot_" +
              symbol+"_"+interval+"_"+start+"_"+end+".csv")


#take_spot_data("BTC", "1m", "2022-01-01", "2022-01-22")
# %%
# take data from binance


def take_future_data(symbol="BTC", interval="1m", start="2022-01-01", end="2022-01-31"):
    # take data from binance
    data = client.futures_historical_klines(
        symbol=symbol+"USDT",
        interval=interval,
        start_str=start,  # 2021-01-01
        end_str=end
    )

    df = pd.DataFrame(data, columns=['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time',
                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df.to_csv(directory_of_python_script +
              "/data/future_"+symbol+"_"+interval+"_"+start+"_"+end+".csv", sep=',')
    # df_2020 = pd.DataFrame(data_2020, columns=['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time',
    #                                           'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    # df_2020.to_csv(directory_of_python_script +
    #               "/data/"+symbol+"_kline_"+interval+"_200101_201231.csv", sep=',')


take_future_data("BTC", "15m", "2021-01-01", "2021-12-31")

# %%
