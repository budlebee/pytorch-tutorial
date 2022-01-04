# add percentage increase or decrease to a csv data.


import pandas as pd
import os


directory_of_python_script = os.path.dirname(os.path.abspath(__file__))


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
