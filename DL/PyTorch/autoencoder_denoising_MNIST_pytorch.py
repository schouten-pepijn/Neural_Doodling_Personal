import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

#%% DATA LOAD
directory = Path("/run/media/pepijn/37ac6286-af0a-4bfb-890f-09f2e9eca83e" \
                 "/Datasets/MNIST_IMG/")
    
transforms = torchvision.transforms.Compose({
    torchvision.transforms.ToTensor()})
    
train_data = torchvision.datasets.MNIST(root=directory,
                                        train=True,
                                        download=True,
                                        transform=transforms).data

test_data = torchvision.datasets.MNIST(root=directory,
                                       train=False,
                                       download=True,
                                       transform=transforms).data

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_train_data = scaler.fit_transform(
    X=train_data.reshape(train_data.shape[0], -1), y=None)
scaled_train_data = torch.tensor(scaled_train_data).float()

scaled_test_data = scaler.transform(
    X=test_data.reshape(test_data.shape[0], -1))
scaled_test_data = torch.tensor(scaled_test_data).float()

#%% CREATE DL MODEL
def createTheMNISTAE(in_shape):
    
    class aenet(nn.Module):
        def __init__(self, in_shape):
            super().__init__()
            
            self.input = nn.Linear(in_shape, 250)
            self.enc = nn.Linear(250, 50)
            self.lat = nn.Linear(50, 250)
            self.dec = nn.Linear(250, in_shape)
            
        def forward(self, x):
            x = F.relu(self.input(x))
            x = F.relu(self.enc(x))
            x = F.relu(self.lat(x))
            x = torch.sigmoid(self.dec(x))
            return x
            
    net = aenet(in_shape)
    
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    return net, criterion, optimizer


net, criterion, optimizer = createTheMNISTAE(scaled_train_data.shape[-1])

#%% TRAIN MODEL
def trainModel(train_data):
    epochs = 20000
    
    batch_size, data_size = train_data.shape
    
    net, criterion, optimizer = createTheMNISTAE(data_size)

    losses = torch.zeros(epochs)

    net.train()
    for epoch in range(epochs):
        
        random_idx = np.random.choice(batch_size, size=64)
        X = train_data[random_idx, :]
        
        yHat = net(X)
        loss = criterion(yHat, X)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses[epoch] = loss.item()
        
        if epoch % 100 == 0:
            print(f'Epoch: {epoch:5} | Loss: {losses[epoch]:.5f}')
        
    return losses, net
        
losses, net = trainModel(scaled_train_data)

#%% EVALUATION
print(f'Final loss: {losses[-1]:.4f}')
plt.figure()
plt.plot(losses, '-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Final loss: {losses[-1]:.4f}')
plt.show()

# test data
X_test_sample = scaled_test_data[:5, :]
with torch.inference_mode():
    yHat_sample = net(X_test_sample)

fig, axs = plt.subplots(2, 5)
for i in range(5):
    axs[0, i].imshow(X_test_sample[i, :].view(28, 28), cmap='gray')
    axs[1, i].imshow(yHat_sample[i, :].view(28, 28), cmap='gray')
    axs[0, i].set_xticks([]), axs[0, i].set_yticks([])
    axs[1, i].set_xticks([]), axs[1, i].set_yticks([])
plt.show()

#%% noisy test data
X_noisy_sample = X_test_sample + torch.rand_like(X_test_sample) / 8
X_noisy_sample[X_noisy_sample > 1] = 1
with torch.inference_mode():
    yHat_denoised = net(X_noisy_sample)

fig, axs = plt.subplots(2, 5)
for i in range(5):
    axs[0, i].imshow(X_noisy_sample[i, :].view(28, 28), cmap='gray')
    axs[1, i].imshow(yHat_denoised[i, :].view(28, 28), cmap='gray')
    axs[0, i].set_xticks([]), axs[0, i].set_yticks([])
    axs[1, i].set_xticks([]), axs[1, i].set_yticks([])
plt.show()