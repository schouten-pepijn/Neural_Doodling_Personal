import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


#%% DATA
def create_linear_data(slope, offset, data_amount, noise_factor):
    x = torch.arange(0, data_amount).float() / data_amount
    y = slope * x + offset
    noise = torch.randn(x.size()) * noise_factor
    y += noise
    
    x, y = x.unsqueeze(1), y.unsqueeze(1)
    return x, y


def create_sine_data(amplitude, phase, data_amount, noise_factor):
    x = torch.arange(0, data_amount).float() / data_amount
    y = amplitude * torch.sin(x * np.pi - phase)
    noise = torch.randn(x.size()) * noise_factor
    y += noise
    
    x, y = x.unsqueeze(1), y.unsqueeze(1)
    return x, y
    

slope = 1.3
offset = 3
data_amount = 600
noise_factor = 0.05


# x, y = create_linear_data(slope, offset, data_amount, noise_factor)

x, y = create_sine_data(slope, offset, data_amount, noise_factor)

plt.figure()
plt.scatter(x, y, s=2)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=42)

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

batch_size= 32
train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                          drop_last=True)

x_sample, y_sample = next(iter(train_loader))
print(f'x shape: {x_sample.shape}, y_shape: {y_sample.shape}')


#%% MODEL
class LinearNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        # self.fc = nn.Linear(input_size, output_size)
        
        self.fc = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.out = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        return x
    
model = LinearNet(input_size=1, output_size=1)

print(model)


#%% TRAINING
def train_model(train_loader, epochs=600):
    input_size = train_loader.dataset.tensors[0].size(1)
    model = LinearNet(input_size=input_size, output_size=1)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        
        losses = 0.
        
        for x_train, y_train in train_loader:
            y_train_pred = model(x_train)
            
            loss = criterion(y_train_pred.squeeze(),
                             y_train.squeeze())
    
            
            losses += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch} Train loss: {losses:.4f}')
    
    return model
    
model = train_model(train_loader)

#%% PREDICTION
model.eval()
x_train = train_set.tensors[0]
y_train = train_set.tensors[1]

with torch.inference_mode():
    y_train_pred = model(x_train)
    y_test_pred = model(x_test)
        
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].scatter(x_train, y_train, c='green', s=4)
ax[0].plot(x_train, y_train_pred, c='red')
ax[1].scatter(x_test, y_test, c='green', s=4)
ax[1].plot(x_test, y_test_pred, c='red')
plt.show()


#%% """ FORECASTING """
# DATA
seq_length = 40
sample_length = len(y) - seq_length - 1
sequences = torch.zeros((sample_length, seq_length))
targets = torch.zeros((sample_length, 1))

for i in range(sample_length):
    sequences[i] = y[i:seq_length+i].squeeze()
    targets[i] = y[seq_length+i+1]

train_set = TensorDataset(sequences, targets)

batch_size= 32
train_loader = DataLoader(train_set, batch_size=32,
                          shuffle=True, drop_last=True)

model = LinearNet(input_size=seq_length, output_size=1)

model(sequences[0])

model = train_model(train_loader, epochs=800)
    
#%% PREDICTION
model.eval()
with torch.inference_mode():
    predictions = model(sequences)
        
plt.figure()
plt.scatter(targets, predictions)
plt.show()

plt.figure()
plt.scatter(x[:seq_length+1], y[:seq_length+1], s=3, alpha=0.8)
plt.scatter(x[seq_length+1:], predictions, s=3, alpha=0.8)
plt.show()

start_seq = sequences[0]

pred_on_pred = torch.zeros(*predictions.shape)
for i in range(sample_length):
    with torch.inference_mode():
        pred_on_pred[i] = model(start_seq)
    start_seq = start_seq.roll(-1)
    start_seq[-1] = pred_on_pred[i]

plt.figure()
plt.scatter(x[:seq_length+1], y[:seq_length+1], s=3, alpha=0.8)
plt.scatter(x[seq_length+1:], pred_on_pred, s=3, alpha=0.8)
plt.show()
