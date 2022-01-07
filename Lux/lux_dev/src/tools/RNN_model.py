#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web
import datetime as dt

start_ = dt.datetime(2021,1,1)
end_ = dt.datetime(2022,1,1)

info_fetched = web.DataReader('YAR.OL', 'yahoo', start_, end_)

training_set = info_fetched['Adj Close'].values
#training_set = pd.read_csv('shampoo.csv')
print(training_set)
#%%

#plt.plot(training_set, label = 'Shampoo Sales Data')
plt.plot(training_set, label = 'Airline Passangers Data')
plt.show()
# %%


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc = MinMaxScaler(feature_range=(-1, 1))
training_set = sc.fit_transform(training_set.reshape(-1, 1))

seq_length = 4
x, y = sliding_windows(training_set.reshape(-1, 1), seq_length)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = torch.Tensor(np.array(x)).to('cuda')
dataY = torch.Tensor(np.array(y)).to('cuda')

trainX = torch.Tensor(np.array(x[0:train_size])).to('cuda')
trainY = torch.Tensor(np.array(y[0:train_size])).to('cuda')

testX = torch.Tensor(np.array(x[train_size:len(x)])).to('cuda')
testY = torch.Tensor(np.array(y[train_size:len(y)])).to('cuda')
# %%

print(train_size)
print(test_size)

print(trainY.shape)
print(trainX.shape)


plt.plot(trainX.cpu().numpy()[:, 0, 0], label = 'x')
plt.plot(trainY.cpu().numpy(), label = 'y')
plt.legend()
plt.show()
# %%

input_size = 1
hidden_size = 100
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
num_epochs = 2000
learning_rate = 0.01
lstm = LSTM(num_classes, input_size, hidden_size, num_layers).to('cuda')

#%%

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

lstm.train()
# Train the model
for epoch in range(num_epochs):
    # trainX_input = trainX
    print(trainX.shape)
    # trainX_input = trainX_input.view(trainX_input.shape[0], 4 , input_size)
    outputs = outputs = lstm(trainX)
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs[:, 0].view(-1, 1), trainY)
    
    loss.backward()
    
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
# %%

lstm.eval()
train_predict = lstm(testX)

print(train_predict.shape)
#%%

data_predict = train_predict.data.cpu().numpy()
dataY_plot = testY.data.cpu().numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)



print(data_predict.shape)
print(dataY_plot.shape)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(np.roll(dataY_plot, 1))
plt.plot(data_predict, label = 'prediction')
# plt.xlim([240, 260])
plt.legend()
plt.suptitle('Time-Series Prediction')
plt.show()
# %%

# %%

# %%

# %%

# %%
