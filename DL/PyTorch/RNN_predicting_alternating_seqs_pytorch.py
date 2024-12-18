import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys

#%% DATA
input_size = 1
num_hidden = 5
num_layers = 1
seq_length = 9
batch_size = 1
num_epochs = 30

N = 150

data = torch.rand(N)
data[::2] *= -1

plt.figure()
plt.plot([-1, N+1], [0,0], '--', color=[0.8,0.8,0.8])
plt.plot(data, 'ks-', markerfacecolor='w')
plt.xlim([-1, N+1])
plt.title("Data stream")
plt.show()


#%% CREATE A RNN MODEL
# many-to-one model
class RNNnet(nn.Module):
    def __init__(self, input_size, num_hidden, num_layers, batch_size,
                 print_toggle=True):
        super().__init__()
        
        self.print_toggle = print_toggle
        
        self.batch_size = batch_size
        
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, num_hidden, num_layers)
        
        self.out = nn.Linear(num_hidden, 1)
        
    def forward(self, x):
        
        if self.print_toggle: print(f"Input: {list(x.shape)}")
        
        hidden = torch.zeros(self.num_layers, self.batch_size,
                             self.num_hidden)
        if self.print_toggle: print(f"Hidden: {list(hidden.shape)}")
        
        y, hidden = self.rnn(x, hidden)
        if self.print_toggle: print(f"RNN-Output: {list(y.shape)}")
        if self.print_toggle: print(f"RNN-Hidden: {list(hidden.shape)}")
        
        o = self.out(y)
        if self.print_toggle: print(f"FNN-Output: {list(o.shape)}")
        
        return o, hidden

net = RNNnet(input_size, num_hidden, num_layers, batch_size, print_toggle=True)
print(net)

X = torch.rand(seq_length, batch_size, input_size)
y = torch.rand(seq_length, batch_size, 1)
y_hat, h = net(X)
print(f"Shapes equal? {y_hat.shape == y.shape}")


#%% TRAIN THE MODEL
net = RNNnet(input_size, num_hidden, num_layers, batch_size, print_toggle=False)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

losses = np.zeros(num_epochs)
sign_acc = np.zeros(num_epochs)

for epoch in range(num_epochs):
    
    seg_losses = []
    seg_acc = []
    hidden_state = torch.zeros(num_layers, batch_size, num_hidden)
    
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
        
        # sign acc
        true_sign = np.sign(torch.squeeze(y).numpy())
        pred_sign = np.sign(torch.squeeze(final_value).detach().numpy())
        accuracy = 100*(true_sign == pred_sign)
        seg_acc.append(accuracy)
    
    losses[epoch] = np.mean(seg_losses)
    sign_acc[epoch] = np.mean(seg_acc)
    
    msg = f"Finished epoch {epoch+1}/{num_epochs}"
    sys.stdout.write("\r" + msg)
    
    
#%% EVALUATION
fig, ax = plt.subplots(1, 2, dpi=200, figsize=(16,5), tight_layout=True)

ax[0].plot(losses, 's-')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].set_title("Model loss")

ax[1].plot(sign_acc, 'm^-', markerfacecolor='g', markersize=14)
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].set_title(f"Sign accuracy (final accuracy: {sign_acc[-1]:.2f}%)")
plt.show()