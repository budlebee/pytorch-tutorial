# add percentage increase or decrease to a csv data.

# %%
import pandas as pd
import os


directory_of_python_script = os.path.dirname(os.path.abspath(__file__))

# %%


def add_pnl(sym, itv):

    train_df = pd.read_csv(directory_of_python_script +
                           "/data/"+sym+"_kline_"+itv+"_210101_211231.csv")

    open_price = train_df['open_price']
    open_price_pnl = []
    for i in range(len(open_price)):
        if i == len(open_price)-1:
            open_price_pnl.append(0)
        else:
            open_price_pnl.append((open_price[i+1]/open_price[i]-1)*100)
    train_df["open_price_pnl"] = open_price_pnl

    close_price = train_df['close_price']
    close_price_pnl = []
    for i in range(len(close_price)):
        if i == len(close_price)-1:
            close_price_pnl.append(0)
        else:
            close_price_pnl.append((close_price[i+1]/close_price[i]-1)*100)
    train_df["close_price_pnl"] = close_price_pnl

    high_price = train_df['high_price']
    high_price_pnl = []
    for i in range(len(high_price)):
        if i == len(high_price)-1:
            high_price_pnl.append(0)
        else:
            high_price_pnl.append((high_price[i+1]/high_price[i]-1)*100)
    train_df["high_price_pnl"] = high_price_pnl

    low_price = train_df['low_price']
    low_price_pnl = []
    for i in range(len(low_price)):
        if i == len(low_price)-1:
            low_price_pnl.append(0)
        else:
            low_price_pnl.append((low_price[i+1]/low_price[i]-1)*100)
    train_df["low_price_pnl"] = low_price_pnl

    open_to_high_diff = []
    for i in range(len(high_price)):
        if i == len(high_price)-1:
            open_to_high_diff.append(0)
        else:
            open_to_high_diff.append(
                ((high_price[i+1]-open_price[i+1])/open_price[i+1])*100)
    train_df["open_to_high_diff"] = open_to_high_diff

    open_to_low_diff = []
    for i in range(len(high_price)):
        if i == len(high_price)-1:
            open_to_low_diff.append(0)
        else:
            open_to_low_diff.append(
                ((low_price[i+1]-open_price[i+1])/open_price[i+1])*100)
    train_df["open_to_low_diff"] = open_to_low_diff

    train_df.to_csv(directory_of_python_script +
                    "/data/"+sym+"_kline_"+itv+"_210101_211231.csv")


add_pnl("BTC", "1minute")
