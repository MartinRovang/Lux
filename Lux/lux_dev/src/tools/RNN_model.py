#%%
from importlib.metadata import requires
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web
import datetime as dt
import pickle
from sklearn.model_selection import train_test_split
import einops
# start_ = dt.datetime(2021,1,1)
# end_ = dt.datetime(2022,1,1)

# info_fetched = web.DataReader('YAR.OL', 'yahoo', start_, end_)

# training_set = info_fetched['Adj Close'].values
#training_set = pd.read_csv('shampoo.csv')
# print(training_set)

df = pickle.load(open('../../database/stockprices/all_stock_vals(2022, 1, 6).pkl', 'rb'))

df.shape
print(df['2020.OL'])
# 247 signal
#%%
train, val = train_test_split(df, test_size=0.2)
print(train.shape)
print(val.shape)
#%%
#%%
#plt.plot(training_set, label = 'Shampoo Sales Data')
train['2020.OL'].plot()
plt.show()
# %%


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data.iloc[i:(i+seq_length)]
        _y = data.iloc[i+seq_length]
        x.append(_x.values)
        y.append(_y.values)

    return np.array(x),np.array(y)

# sc = MinMaxScaler(feature_range=(-1, 1))
# training_set = sc.fit_transform(training_set.reshape(-1, 1))

seq_length = 30
x_train, y_train = sliding_windows(train, seq_length)
x_val, y_val = sliding_windows(val, seq_length)


#%%
print(x_train.shape)
plt.plot(x_train[7, :, 0])
plt.show()
#%%
# train_size = int(len(y) * 0.67)
# test_size = len(y) - train_size

trainX = torch.Tensor(x_train)
trainY = torch.Tensor(y_train)

valX = torch.Tensor(x_val)
valY = torch.Tensor(y_val)

#%%
# dataX = torch.Tensor(np.array(x)).to('cuda')
# dataY = torch.Tensor(np.array(y)).to('cuda')

# trainX = torch.Tensor(np.array(x[0:train_size])).to('cuda')
# trainY = torch.Tensor(np.array(y[0:train_size])).to('cuda')

# testX = torch.Tensor(np.array(x[train_size:len(x)])).to('cuda')
# testY = torch.Tensor(np.array(y[train_size:len(y)])).to('cuda')
# %%

print(trainY.shape)
print(trainX.shape)


plt.plot(trainX.cpu().numpy()[:, 0, 0], label = 'x')
plt.plot(trainY.cpu().numpy()[:, 0], label = 'y')
plt.legend()
plt.show()
# %%

input_size = 1
hidden_size = 150
num_layers = 1

num_classes = 1

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to('cuda')
        
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to('cuda')
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        # print(ula.shape)

        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        
        return out
# %%
num_epochs = 200
learning_rate = 0.01
lstm = LSTM(num_classes, input_size, hidden_size, num_layers).to('cuda')

#%%

#%%
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
batchsize = 5
# LSTM INPUT N,L,H 
trainX = einops.rearrange(trainX, 'n_time sequence_length n_stocks -> n_stocks sequence_length n_time')
trainY = einops.rearrange(trainY, 'n_time n_stocks -> n_stocks n_time')

valX = einops.rearrange(valX, 'n_time sequence_length n_stocks -> n_stocks sequence_length n_time')
valY = einops.rearrange(valY, 'n_time n_stocks -> n_stocks n_time')
lstm.train()
# Train the model
for epoch in range(num_epochs):
    # trainX_input = trainX
    # print(trainX.shape)
    # print(trainY.shape)
    lstm.train()
    for train_ in range(0, trainX.shape[0]):
        perm_stonks = torch.randperm(trainX.size(0))
        perm_time = torch.randperm(trainX.size(2))[0]
        idx_stonks = perm_stonks[:batchsize]
        batch_trainx = trainX[idx_stonks, :, perm_time][:, :, None]
        batch_trainy = trainY[idx_stonks,  perm_time][:, None]
        # trainX_input = trainX_input.view(trainX_input.shape[0], 4 , input_size)
        batch_trainx = batch_trainx.to('cuda')
        batch_trainy = batch_trainy.to('cuda')
        outputs = lstm(batch_trainx)
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(outputs, batch_trainy)
        
        loss.backward()
        optimizer.step()
        # if epoch % 100 == 0:
        #     print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    
    lstm.eval()
    with torch.no_grad():
        for val_ in range(0, valX.shape[0]):
            perm_stonks = torch.randperm(valX.size(0))
            perm_time = torch.randperm(valX.size(2))[0]
            idx_stonks = perm_stonks[:batchsize]
            batch_valx = valX[idx_stonks, :, perm_time][:, :, None]
            batch_valy = valY[idx_stonks,  perm_time][:, None]
            # trainX_input = trainX_input.view(trainX_input.shape[0], 4 , input_size)
            batch_valx = batch_valx.to('cuda')
            batch_valy = batch_valy.to('cuda')
            outputs = lstm(batch_valx)
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, batch_valy)

            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, validation loss: %1.5f" % (epoch, loss.item()))

#%%
lstm.eval()
val_predict = lstm(batch_valx).detach().cpu().numpy()
batch_valx_ = batch_valx.detach().cpu().numpy()
batch_valy_ = batch_valy.detach().cpu().numpy()
print(batch_valx_.shape)
print(batch_valx_.shape)

plt.plot(batch_valx_[0, :, 0])
plt.plot(len(batch_valx_[0, :, 0]) + 1 , val_predict[0, :], 'x')
plt.plot(len(batch_valx[0, :, 0]) + 1 , batch_valy_[0, :], 'x')
plt.show()



# %%

test = df['AFG.OL'].values
N = len(test)

test2 = test[:N-30]
test_future = test[-30:]
test2_for_pred = torch.Tensor(test2[-30:])[None, :, None]

print(test2_for_pred.shape)


plt.plot(test2_for_pred[0, :, 0].detach().cpu().numpy())
plt.show()
plt.plot(test_future)
plt.show()

#%%

print(test_future.shape)
#%%
lstm.eval()
val_predict = lstm(test2_for_pred.to('cuda')).detach().cpu().numpy()
plt.plot(test2[-30:])
plt.show()
plt.plot(test_future)
plt.plot(1, val_predict, 'x')
plt.show()

# %%

# %%

# %%

# %%
