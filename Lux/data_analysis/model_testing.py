#%%
import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = torch.linspace(0,799, 800)
y = torch.sin(x * 2 * np.pi / 40)


plt.figure(figsize = (12,4))
plt.xlim(-10, 801)
plt.grid(True)
plt.plot(y.numpy())
# %%
test_size = 40
train_set = y[:-test_size]
test_set = y[-test_size:]
plt.figure(figsize = (12,4))
plt.xlim(-10, 801)
plt.grid(True)
plt.plot(train_set.numpy())
# %%

def input_data(seq, ws):
  output = []   
  L = len(seq)
  for i in range((L) - ws):
    window = seq[i:i+ws]
    label = seq[i+ws:i+ws+1]
    print(i)
    output.append((window, label))
 
  return output
# %%
window_size = 40
train_data = input_data(train_set, window_size)
print(train_data)
# %%
plt.stem(train_data[0][0])
# %%
class myLSTM(nn.Module):
  def __init__(self, input_size=1, hidden_size=50, out_size=1):
    super().__init__()
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(input_size, hidden_size )
    self.linear = nn.Linear(hidden_size, out_size)
    self.hidden = (torch.zeros(1,1,hidden_size ) , torch.zeros(1,1,hidden_size)  )
 
  def forward(self, seq):
    lstm_out, self.hidden = self.lstm(seq.view( len(seq),1,-1 ), self.hidden )
    lstm_out = torch.nn.functional.dropout(lstm_out.view(  len(seq) ,-1 ), p=0.25, training=True, inplace=False)
    pred = self.linear(lstm_out)  
    return pred[-1]


# %%
model = myLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
print(model)
# %%
for p in model.parameters():
  print(p.numel())
# %%
epochs = 10
future = 40
for i in range(epochs):
  for seq, y_train in train_data:
    optimizer.zero_grad()
    model.hidden = (torch.zeros(1,1,model.hidden_size) ,
                    torch.zeros(1,1,model.hidden_size))
    
    y_pred = model(seq)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
 
  print(f"Epoch {i} Loss {loss.item()} ")
  preds = train_set[-window_size:].tolist()
 
  for f in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
      model.hidden = (torch.zeros(1,1,model.hidden_size) ,
                      torch.zeros(1,1,model.hidden_size))
      preds.append(model(seq).item())
 
  loss = criterion(torch.tensor(preds[-window_size :]), y[760:] )
  print(f'Performance on test range: {loss}')
# %%
plt.figure(figsize=(12,4))
plt.xlim(700, 801)
plt.grid(True)
plt.plot(y.numpy())
plt.plot(range(760,800), preds[window_size:])
plt.show()
# %%
preds = y[-window_size:].tolist()
for i in range(future):
  seq = torch.FloatTensor(preds[-window_size:])
  with torch.no_grad():
 
    model.hidden = (torch.zeros(1,1,model.hidden_size), 
                    torch.zeros(1,1,model.hidden_size))
    
    preds.append(model(seq).item())
# %%
plt.figure(figsize = (12,4))
plt.xlim(0,841)
plt.grid(True)
plt.plot(y.numpy())
plt.plot(range(800, 800+future), preds[window_size:])
# %%


plt.figure(figsize=(12,4))
# plt.xlim(700, 801)
plt.grid(True)
plt.plot(y.numpy())
ensemble_result = {}

for ensemble_num in range(20):
    preds = test_set[-window_size:].tolist()
    for i in range(0, 10):
        with torch.no_grad():
            seq = torch.FloatTensor(preds[-window_size:])
            model.hidden = (torch.zeros(1,1,model.hidden_size) ,
                            torch.zeros(1,1,model.hidden_size))
            preds.append(model(seq).item())
    ensemble_result[ensemble_num] = preds
    # print(preds)
    plt.plot(range(760,810), preds)
plt.show()

#%%
data_array = np.empty((20, 10))

for i, key in enumerate(ensemble_result):
    data_array[i, :] = ensemble_result[key][-10:]

mean_over_axis = np.mean(data_array, axis=0)
print(mean_over_axis)
std_over_axis = np.std(data_array, axis=0)
print(std_over_axis)


plt.plot(y.numpy())
# Fill between mean and std
plt.plot(range(800, 810), mean_over_axis, color='red', linewidth=1)
plt.fill_between(range(800, 810), mean_over_axis - std_over_axis, mean_over_axis + std_over_axis, alpha=0.5, color = 'black')
plt.show()


# %%
print(test_set)
# %%
