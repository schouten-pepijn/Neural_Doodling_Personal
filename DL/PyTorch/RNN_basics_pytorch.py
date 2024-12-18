import torch
import torch.nn as nn
import numpy as np


#%% MODEL
input_size = 9
hidden_size = 16
num_layers = 1
act_fun = 'tanh'
bias = True

rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
             num_layers=num_layers, nonlinearity=act_fun,
             bias=bias)
print(rnn)


#%% DATA
seq_length = 5
batch_size = 2

X = torch.rand(seq_length, batch_size, input_size)

# Usually a zero vector
hidden = torch.zeros(num_layers, batch_size, hidden_size)


#%% EXAMPLE OUTPUTS
y, h = rnn(X, hidden)

print(f"Input shape: {list(X.shape)}")
print(f"Hidden shape: {list(h.shape)}")
print(f"Output shape: {list(y.shape)}")

y, h1 = rnn(X, hidden)
print(h1, sep="\n")

y, h2 = rnn(X)
print(h2, sep="\n")

print(h1-h2)


#%% Check the learned parameters
for p in rnn.named_parameters():
    if 'weight' in p[0]:
        print(f"{p[0]} has size {list(p[1].shape)}")
        

#%% CREATE A RNN MODEL
# many-to-one model
class RNNnet(nn.Module):
    def __init__(self, input_size, num_hidden, num_layers, print_toggle=True):
        super().__init__()
        
        self.print_toggle = print_toggle
        
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, num_hidden, num_layers)
        
        self.out = nn.Linear(num_hidden, 1)
        
    def forward(self, x):
        
        if self.print_toggle: print(f"Input: {list(x.shape)}")
        
        hidden = torch.zeros(self.num_layers, batch_size, self.num_hidden)
        if self.print_toggle: print(f"Hidden: {list(hidden.shape)}")
        
        y, hidden = self.rnn(x, hidden)
        if self.print_toggle: print(f"RNN-Output: {list(y.shape)}")
        if self.print_toggle: print(f"RNN-Hidden: {list(hidden.shape)}")
        
        o = self.out(y)
        if self.print_toggle: print(f"FNN-Output: {list(o.shape)}")
        
        return o, hidden
        
input_size = 9
num_hidden = 16
num_layers = 1        
rnn = RNNnet(input_size, num_hidden, num_layers, print_toggle=True)

print(rnn, sep="\n")
       
for p in rnn.named_parameters():
    print(f"{p[0]} has size {list(p[1].shape)}")
    

#%% TEST with dummy data
X = torch.rand(seq_length, batch_size, input_size)
y = torch.rand(seq_length, batch_size, 1)
y_hat, h = rnn(X)

print(f"Shapes equal? {y_hat.shape == y.shape}")


criterion = nn.MSELoss()
loss = criterion(y_hat, y)
print(loss)


