# add percentage increase or decrease to a csv data.

# %%
import pandas as pd
import os
from math import log
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 

directory_of_python_script = os.path.dirname(os.path.abspath(__file__))

# %%


def add_pnl(sym, itv, start, end):

    train_df = pd.read_csv(directory_of_python_script +
                           "/data/"+sym+"_kline_"+itv+"_"+start+"_"+end+".csv")

    volume = train_df['volume']

    volume_norm = []
    for i in range(len(volume)):
        volume_norm.append((volume[i])/max(volume))

    open_price = train_df['open_price']
    open_price_pnl = []
    for i in range(len(open_price)):
        if i == len(open_price)-1:
            open_price_pnl.append(0)
        else:
            open_price_pnl.append((open_price[i+1]/open_price[i]-1)*100)
    train_df["open_price_pnl"] = open_price_pnl
    open_price_log = []
    for i in range(len(open_price)):
        if i == len(open_price)-1:
            open_price_log.append(0)
        else:
            open_price_log.append(log(open_price[i+1])-log(open_price[i]))
    train_df["open_price_log"] = open_price_log

    close_price = train_df['close_price']
    close_price_pnl = []
    #arr_scaled = scaler.fit_transform(df) 
    
    for i in range(len(close_price)):
        if i == len(close_price)-1:
            close_price_pnl.append(0)
        else:
            close_price_pnl.append((close_price[i+1]/close_price[i]-1)*100)
    train_df["close_price_pnl"] = close_price_pnl
    close_price_log = []
    for i in range(len(close_price)):
        if i == len(close_price)-1:
            close_price_log.append(0)
        else:
            close_price_log.append(log(close_price[i+1])-log(close_price[i]))
    train_df["close_price_log"] = close_price_log

    high_price = train_df['high_price']
    high_price_pnl = []
    for i in range(len(high_price)):
        if i == len(high_price)-1:
            high_price_pnl.append(0)
        else:
            high_price_pnl.append((high_price[i+1]/high_price[i]-1)*100)
    train_df["high_price_pnl"] = high_price_pnl
    high_price_log = []
    for i in range(len(high_price)):
        if i == len(high_price)-1:
            high_price_log.append(0)
        else:
            high_price_log.append(log(high_price[i+1])-log(high_price[i]))
    train_df["high_price_log"] = high_price_log

    low_price = train_df['low_price']
    
    low_price_pnl = []
    for i in range(len(low_price)):
        if i == len(low_price)-1:
            low_price_pnl.append(0)
        else:
            low_price_pnl.append((low_price[i+1]/low_price[i]-1)*100)
    train_df["low_price_pnl"] = low_price_pnl
    low_price_log = []
    low_price_log_norm = []
    low_price_log_per_volume = []
    for i in range(len(low_price)):
        if i == len(low_price)-1:
            low_price_log.append(0)
        else:
            low_price_log.append(log(low_price[i+1])-log(low_price[i]))
    low_price_log_max = max(low_price_log) if max(low_price_log)>abs(min(low_price_log)) else abs(min(low_price_log))
    for i in range(len(low_price_log)):
        low_price_log_norm.append(low_price_log[i]/low_price_log_max)
        factor = volume_norm[i] if volume_norm[i] != 0 else 1
        low_price_log_per_volume.append(low_price_log[i]*factor)
    train_df["low_price_log"] = low_price_log
    train_df["low_price_log_norm"] = low_price_log_norm
    train_df["low_price_log_per_volume"]=low_price_log_per_volume

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
                           "/data/"+sym+"_kline_"+itv+"_"+start+"_"+end+".csv")


add_pnl("BTC", "15m","2021-01-01","2022-01-07")

#%%

filedir = "/data/"+"BTC"+"_kline_"+"15m"+"_"+"2021-01-01"+"_"+"2022-01-07"+".csv"
df = pd.read_csv(directory_of_python_script + filedir)

volume = df['volume']
volume_norm = []

high_price = df['high_price']
low_price = df['low_price']
open_price = df['open_price']
close_price = df['close_price']

for i in range(len(volume)):
    volume_norm.append((volume[i])/max(volume))

# 변동성을 기준으로 보면 어떨까? 해당 분봉에서 high-low 값이랑, open-close 값을 볼륨으로 나눈 것을 보자. 
#%%
diff_high_low_per_volume = []
diff_open_close_per_volume = []
for i in range(len(volume)):
    if(volume[i]==0):
        diff_high_low_per_volume.append(0)
        diff_open_close_per_volume.append(0)
    else:
        diff_high_low_per_volume.append((high_price[i]-low_price[i])/volume[i])
        diff_open_close_per_volume.append((open_price[i]-close_price[i])/volume[i])

df["diff_high_low_per_volume"] = diff_high_low_per_volume
df["diff_open_close_per_volume"] = diff_open_close_per_volume

#%%

diff_percent_high_low_per_volume = []
diff_percent_open_close_per_volume = []
for i in range(len(volume)):
    if(volume[i]==0):
        diff_percent_high_low_per_volume.append(0)
        diff_percent_open_close_per_volume.append(0)
    else:
        diff_percent_high_low_per_volume.append(volume_norm[i]*(high_price[i]-low_price[i])/low_price[i])
        diff_percent_open_close_per_volume.append(volume_norm[i]*(open_price[i]-close_price[i])/close_price[i])

df["diff_percent_high_low_per_volume"] = diff_percent_high_low_per_volume
df["diff_percent_open_close_per_volume"] = diff_percent_open_close_per_volume


df.to_csv(directory_of_python_script +filedir)
# %%
