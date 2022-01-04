# %%
import pandas as pd
import os



directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(directory_of_python_script +
                           "/data/"+"BTC"+"_kline_"+"1minute"+"_210101_211231.csv")

#%%
def rtn_timing(df):
    print("returning buy/sell timing")