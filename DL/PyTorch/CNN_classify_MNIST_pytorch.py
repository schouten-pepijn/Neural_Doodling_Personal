import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchinfo import summary

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import numpy as np
import matplotlib.pyplot as plt

import random

# Variables
print_toggle = True
n_epochs = 10
batch_size = 32

#%% data import
train_data = torchvision.datasets.MNIST(root='./data', train=True,
                                                      download=True, transform=T.ToTensor())
test_data = torchvision.datasets.MNIST(root='./data', train=False,
                                                    download=True, transform=T.ToTensor())

train_dataloader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=test_data.data.size(0),
                             shuffle=False)

#%% create model
def createMNISTNet(print_toggle=False):
    class mnistNet(nn.Module):
        def __init__(self, print_toggle):
            super().__init__()

            self.conv1 =  nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=1)
            self.conv2 =  nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)

            self.fc1 = nn.Linear(180, 50)
            self.out = nn.Linear(50, 10)

            self.print = print_toggle

        def forward(self, x):

            if self.print:
                print(f"Input: {x.shape}")

            x = F.relu(F.max_pool2d(self.conv1(x), 2))

            if self.print:
                print(f"Layer conv1/pool1: {x.shape}")

            x = F.relu(F.max_pool2d(self.conv2(x), 3))

            if self.print:
                print(f"Layer conv2/pool2: {x.shape}")

            n_units = x.shape.numel() // x.shape[0]
            x = x.view(-1, n_units)

            if self.print:
                print(f"Flattened view: {x.shape}")

            x = F.relu(self.fc1(x))

            if self.print:
                print(f"Layer fc1: {x.shape}")

            x = self.out(x)

            if self.print:
                print(f"Layer out: {x.shape}")

            return x

    net = mnistNet(print_toggle)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    return net, criterion, optimizer


#%% test model
net, criterion, optimizer = createMNISTNet(print_toggle)

X, y = next(iter(train_dataloader))
y_hat = net(X)

print('  ')
print(y_hat.shape)
print(y.shape)

loss = criterion(y_hat, y)

print('  ')
print(f"loss: {loss}")

summary(net, (1, 1, 28, 28))

#%% train model
def trainModel(n_epochs):

    net, criterion, optimizer = createMNISTNet()

    losses = torch.zeros(n_epochs)
    train_acc = []
    test_acc = []

    for epoch in range(n_epochs):
        net.train()
        batch_acc = []
        batch_loss = []

        for X, y in train_dataloader:

            y_hat = net(X)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

            matches = torch.argmax(y_hat, 1) == y
            batch_acc.append(matches.float().mean().item())

        train_acc.append(np.mean(batch_acc))
        losses[epoch] = np.mean(batch_loss)

        net.eval()
        X, y = next(iter(test_dataloader))
        with torch.inference_mode():
            y_hat = net(X)

        test_acc.append((torch.argmax(y_hat, 1) == y).float().mean().item())

        if epoch % 2 == 0:
            print(f"Epoch: {epoch} | Train acc: {train_acc[-1]:.4f} | Test acc: {test_acc[-1]:.4f}")

    return train_acc, test_acc, losses, net


train_acc, test_acc, losses, net = trainModel(n_epochs)


#%% evaluate model
fig, ax = plt.subplots(1, 2, figsize=(16, 5), dpi=200)
ax[0].plot(losses, 's-')
ax[1].plot(train_acc, 's-', label='Train')
ax[1].plot(test_acc, 's-', label='Test')

ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].legend()
plt.show()



