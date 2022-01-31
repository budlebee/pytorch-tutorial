from torch_tut import plot_prdt, seq_data
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

dir_of_this = "/Users/zowan/Documents/python/binance-pytorch"

model = torch.load(dir_of_this+"/models/rnn_21_1m.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

df = pd.read_csv(dir_of_this + "/data/" +
                 "BTC"+"_kline_"+"1m"+"_2022-01-01_2022-01-05.csv")

scaler = MinMaxScaler()
df[['open_price', 'high_price', 'low_price', 'close_price', 'volume']] = scaler.fit_transform(
    df[['open_price', 'high_price', 'low_price', 'close_price', 'volume']])
df.info()
X = df[['open_price', 'high_price', 'low_price', 'volume']].values
y = df['close_price'].values


seq_len = 5

x_seq, y_seq = seq_data(X, y, seq_len)

x_test_seq = x_seq
y_test_seq = y_seq
print(x_test_seq.size(), y_test_seq.size())


with torch.no_grad():
    with torch.no_grad():
        train_prdt = []
        test_prdt = []
        for data in test_loader:
            seq, target = data
            out = model(seq)
            test_prdt += out.cpu().numpy().tolist()

    total = train_prdt+test_prdt
    plt.figure(figsize=(20, 10))
    plt.plot(np.ones(100)*len(train_prdt),
             np.linspace(0, 1, 100), '--', linewidth=0.6)
    plt.plot(actual, '--')
    plt.plot(total, 'b', linewidth=0.6)
    plt.legend(['train', 'actual', 'prediction'])
    plt.show()
