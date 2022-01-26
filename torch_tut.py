# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

#directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
directory_of_python_script = "/Users/zowan/Documents/python/binance-pytorch"
# %% 학습에 사용할 CPU나 GPU 장치를 얻습니다.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# %%


class CustomDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.data = []  # 데이터를 정제해 담을 행렬 생성
        df = pd.read_csv(dir)  # pandas를 이용한 csv 파일 읽기
        for idx in df.index:
            temp_target = df.values[idx][0]  # 타겟 행 받아오기
            temp_input = df.values[idx][1:]  # 행렬 행 받아오기
            # 타겟 / 행렬 구분된 데이터를 data에 첨부
            self.data.append((temp_input, temp_target))
        self.transform = transform  # 데이터 받아오기 이후 데이터 전처리

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x


# %%

file_name = "BTC_kline_15m_2021-01-01_2022-01-07.csv"

df = pd.read_csv(directory_of_python_script + "/data/" +
                 file_name)
scaler = MinMaxScaler()
df[['open_price_log', 'high_price_log', 'low_price_log', 'close_price_log', 'volume_per_num_of_trades', 'low_price_log_per_volume']] = scaler.fit_transform(
    df[['open_price_log', 'high_price_log', 'low_price_log', 'close_price_log', 'volume_per_num_of_trades', 'low_price_log_per_volume']])
# df.head()
df.info()
X = df[['open_price_log', 'high_price_log', 'low_price_log',
        'volume_per_num_of_trades', 'low_price_log_per_volume']].values
y = df['close_price_log'].values

# %% make sequence data


def seq_data(x, y, seq_len):
    x_seq = []
    y_seq = []
    for i in range(len(x)-seq_len):
        x_seq.append(x[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1, 1])

# 앞의 절반은 트레인용, 뒤의 절반은 테스트용으로 써보자


#split = round(len(X)/(1.4))
split = len(X)//2
seq_len = 4

x_seq, y_seq = seq_data(X, y, seq_len)

x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]
print(x_train_seq.size(), y_train_seq.size())
print(x_test_seq.size(), y_test_seq.size())

# %%


batch_size = 20

# 데이터로더를 생성합니다.
# train_data = CustomDataset(dir=directory_of_python_script +
#                           "/data/"+"BTC"+"_kline_"+"1h"+"_210101_211231.csv")
# test_data = CustomDataset(dir=directory_of_python_script +
#                          "/data/"+"BTC"+"_kline_"+"1h"+"_200101_201231.csv")
train_data = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test_data = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

input_size = x_seq.size(2)
layer_num = 2
hidden_size = 8

for X, y in test_dataloader:
    print("Shape of X: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


# %% 모델을 정의합니다.


class TestRNN(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, layer_num, device):
        super(TestRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.rnn = nn.RNN(input_size, hidden_size, layer_num, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*seq_len, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.layer_num, x.size()[
                         0], self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


model = TestRNN(input_size=input_size,
                hidden_size=hidden_size,
                seq_len=seq_len,
                layer_num=layer_num,
                device=device).to(device)

loss_fn = nn.MSELoss()
lr = 1e-3
epochs = 200
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %% train
loss_graph = []
n = len(train_dataloader)

for epoch in range(epochs):
    running_loss = 0.0
    for data in train_dataloader:
        seq, target = data  # batch data
        out = model(seq)  # model
        loss = loss_fn(out, target)  # loss by output
        optimizer.zero_grad()  # clear grad
        loss.backward()  # backprop
        optimizer.step()  # update
        running_loss += loss.item()  # sum loss
    loss_graph.append(running_loss/n)  # average loss
    if epoch % 20 == 0:
        print(f"epoch: {epoch}, loss: {running_loss/n}")

# %% check loss graph

plt.figure(figsize=(20, 10))
plt.plot(loss_graph)
plt.show()

# %% check prediction


def plot_prdt(train_loader, test_loader, actual, model):
    with torch.no_grad():
        train_prdt = []
        test_prdt = []
        for data in train_loader:
            seq, target = data
            out = model(seq)
            train_prdt += out.cpu().numpy().tolist()
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


plot_prdt(train_dataloader, test_dataloader,
          df['close_price_log'][seq_len:], model=model)

# %%
torch.save(model, './models/rnn_21-01-01_22-01-07_15m.pth')

# %%
