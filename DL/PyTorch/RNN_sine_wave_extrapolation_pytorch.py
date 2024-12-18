import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys


# %% VARIABLES
N = 800
input_size = 1
num_hidden = 18
num_layers = 1
num_epochs = 20
seq_length = 50
batch_size = 1
rnn_type = 'RNN'


#%% DATA
time = torch.linspace(0, 30*np.pi, N)
data = torch.sin(time + torch.cos(time))

plt.figure(dpi=200, figsize=(15,4))
plt.plot(time, data, 'ks-', markerfacecolor='w')
plt.xlim([-5, time[-1]+4])
plt.show()

plt.figure(dpi=200, figsize=(15,4))
plt.plot(time[:seq_length], data[:seq_length], 'ks-')
plt.show()

#%% MODEL
class RNNnet(nn.Module):
    def __init__(self, input_size, num_hidden, num_layers, batch_size,
                 rnn_type='RNN', print_toggle=True):
        super().__init__()
        
        self.print_toggle = print_toggle
        
        self.batch_size = batch_size
        
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        network = getattr(nn, rnn_type)
        self.rnn = network(input_size, num_hidden, num_layers)
        
        self.out = nn.Linear(num_hidden, 1)
        
    def forward(self, x):
        
        if self.print_toggle: print(f"Input: {list(x.shape)}")
        
        hidden = torch.zeros(self.num_layers, self.batch_size, self.num_hidden)
        if self.print_toggle: print(f"Hidden: {list(hidden.shape)}")
        
        y, hidden = self.rnn(x, hidden)
        if self.print_toggle: print(f"RNN-Output: {list(y.shape)}")
        if self.print_toggle: print(f"RNN-Hidden: {list(hidden.shape)}")
        
        o = self.out(y)
        if self.print_toggle: print(f"FNN-Output: {list(o.shape)}")
        
        return o, hidden.detach()

net = RNNnet(input_size, num_hidden, num_layers, batch_size, print_toggle=True)
print(net)

y, h = net(torch.rand(seq_length, batch_size, input_size))
print(y.shape, h.shape)


#%% TRAINING
net = RNNnet(input_size, num_hidden, num_layers, batch_size,
             rnn_type, print_toggle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)


losses = np.zeros(num_epochs)
sign_acc = np.zeros(num_epochs)

for epoch in range(num_epochs):
    
    seg_losses = []
    
    # loop over segments
    for seq in range(N-seq_length):
        X = data[seq:seq+seq_length].view(seq_length, 1, 1)
        y = data[seq+seq_length].view(1, 1)
        
        # forward pass
        y_hat, hidden_state = net(X)
        final_value = y_hat[-1] # grab only last output value
        loss = criterion(final_value, y)
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # loss
        seg_losses.append(loss.item())

    losses[epoch] = np.mean(seg_losses)
    
    msg = f"Finished epoch {epoch+1}/{num_epochs} | "\
    f"Train loss: {losses[epoch]:.3f}"
    sys.stdout.write("\r" + msg)
    
    
#%% EVALUATION

plt.figure(dpi=200)
plt.plot(losses, 'ks-', label='Training loss')
plt.legend()
plt.show()

y_pred = np.zeros(N)
y_pred[:] = np.nan
net.eval()
for time_i in range(N-seq_length):
    
    X = data[time_i:time_i+seq_length].view(seq_length, 1, 1)
    with torch.inference_mode():
        yy, hh = net(X)
        y_pred[time_i+seq_length] = yy[-1].squeeze().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(16,4))
ax[0].plot(data, 'bs-', label='Actual data', markersize=3)
ax[0].plot(y_pred, 'ro-', label='Predicted data', markersize=3)
ax[0].set_ylim([-1.1, 1.1])
ax[0].legend()

ax[1].plot(data[seq_length:], y_pred[seq_length:], 'mo', markersize=3)
r = np.corrcoef(data[seq_length:], y_pred[seq_length:])
ax[1].set_title(f'r={r[0,1]:.2f}')
plt.tight_layout()
plt.show()


#%% LONGER EXTRAPOLATION
y_pred = torch.zeros(2*N)
y_pred[:N] = data

net.eval()
for time_i in range(2*N-seq_length):
    X = y_pred[time_i:time_i+seq_length].view(seq_length, 1, 1)
    with torch.inference_mode():
        yy, hh = net(X)
    y_pred[time_i+seq_length] = yy[-1]

y_pred = y_pred.squeeze().detach().numpy()


fig, ax = plt.subplots(1, 1, figsize=(16,4))
ax.plot(data, 'bs-', label='Actual data', markersize=3)
ax.plot(y_pred, 'ro-', label='Predicted data', markersize=3)
ax.set_ylim([-1.1, 1.1])
ax.legend()
plt.tight_layout()
plt.show()