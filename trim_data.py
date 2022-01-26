# %% add percentage increase or decrease to a csv data.


import pandas as pd
import os
from tqdm import tqdm

directory_of_python_script = os.path.dirname(os.path.abspath(__file__))

# %%


def trim_data(sym, itv, start_yr, end_yr):

    df = pd.read_csv(directory_of_python_script +
                     "/data/"+sym+"_kline_"+itv+"_"+start_yr+"0101_"+end_yr+"1231.csv", sep=',')

    even = []
    odd = []

    for i in range(len(df)):
        if i % 2 == 0:
            even.append(df.loc[i])
        else:
            odd.append(df.loc[i])

    even_df = pd.DataFrame(even, columns=['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time',
                                          'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    odd_df = pd.DataFrame(odd, columns=['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'close_time',
                                        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

    even_df.to_csv(directory_of_python_script +
                   "/data/"+sym+"_kline_"+itv+"_"+start_yr+"0101_"+end_yr+"1231_even.csv")
    odd_df.to_csv(directory_of_python_script +
                  "/data/"+sym+"_kline_"+itv+"_"+start_yr+"0101_"+end_yr+"1231_odd.csv")


trim_data("BTC", "30m", "17", "21")

# %% make volume bar from 1m data
df = pd.read_csv(directory_of_python_script +
                 "/data/"+"future_BTC_1m_2022-01-01_2022-01-22.csv", sep=',')

# 전체 볼륨을 계산하고, 60분 어치로 쪼갠다.
# 1분봉 데이터가 60만개이면, 볼륨을 전부 더한뒤, 1만으로 나눈것이 볼륨 쓰레스홀드.


def make_volume_bar(df, chunk):
    thr = df['volume'].sum()*chunk / len(df.index)
    df_vbar = pd.DataFrame()
    accum = 0
    close_price = []
    volume = []
    for i in range(len(df)):
        accum += df.loc[i, 'volume']
        if accum > thr:
            volume.append(accum)
            close_price.append(df.loc[i, 'close_price'])
            accum = 0
    df_vbar['volume'] = volume
    df_vbar['close_price'] = close_price
    return df_vbar


df_vbar = make_volume_bar(df, 15)
df_vbar.to_csv(directory_of_python_script +
               '/data/future_volumebar_BTC_1m_2022-01-01_2022-01-22.csv')
print(df_vbar['volume'])

# %% make dollar bar from 1m data

df = pd.read_csv(directory_of_python_script +
                 "/data/"+"future_BTC_1m_2022-01-01_2022-01-22.csv", sep=',')


def make_dollar_bar(df, chunk):
    df_dbar = pd.DataFrame()

    close_price = []
    traded_dollar = []
    # production of volume and close price = actual traded money.
    total_dollar_volume = 0
    for i in range(len(df)):
        dll = df.loc[i, 'close_price']*df.loc[i, 'volume']
        total_dollar_volume += dll
        traded_dollar.append(dll)
    thr = chunk*total_dollar_volume/len(df.index)
    accum = 0
    dollar_bar = []
    close_price = []
    for i in range(len(traded_dollar)):
        accum += traded_dollar[i]
        if accum > thr:
            # accum 이 이전 인덱스보다 너무 크면 이월시키는 로직 추가해야됨.
            dollar_bar.append(accum)
            close_price.append(df.loc[i, 'close_price'])
            accum = 0
    df_dbar['dollar_bar'] = dollar_bar
    df_dbar['close_price'] = close_price
    return df_dbar


df_dbar = make_dollar_bar(df, 15)
df_dbar.to_csv(directory_of_python_script +
               '/data/future_dollarbar_BTC_1m_2022-01-01_2022-01-22.csv')
print(df_dbar['dollar_bar'])

# %%
