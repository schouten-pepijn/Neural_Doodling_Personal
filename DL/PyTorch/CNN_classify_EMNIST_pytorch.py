import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import copy
import time

from sklearn.model_selection import train_test_split
import sklearn.metrics as skm

from torchinfo import summary

import torchvision
from torchvision import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.__version__)
print(torchvision.__version__)
print(f"device: {device}")

#%% data
root = "/home/pepijn/Desktop/EMINST"
data = datasets.EMNIST(root, split='letters',
                             train=True, download=True)

print(f"classes: {data.classes}")
print(f"{len(data.classes)} classes")
print(f"Data size: {data.data.shape}")

images = data.data.view([124800, 1, 28, 28]).float()
print(f"Tensor data: {images.shape}")

# remove N/A class
print(f"N/A count: {torch.sum(data.targets == 0)}")
print(f"Unique classes: {torch.unique(data.targets)}")

categories = data.classes[1:]
labels = copy.deepcopy(data.targets) - 1

print(f"N/A count: {torch.sum(labels == 0)}")
print(f"Unique classes: {torch.unique(labels)}")

#%% normalization
fig = plt.figure(dpi=200)
plt.hist(images[:10,:,:,:].view(1,-1).detach(), 40)
plt.title("Raw values")
plt.show()

images /= torch.max(images)

fig = plt.figure(dpi=200)
plt.hist(images[:10,:,:,:].view(1,-1).detach(), 40)
plt.title("Normalized values")
plt.show()

#%% train test splits
train_data, test_data, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.1)

train_set = TensorDataset(train_data, train_labels)
test_set = TensorDataset(test_data, test_labels)

batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=test_set.tensors[0].shape[0],)

print(f"Train data shape = {train_loader.dataset.tensors[0].shape}")
print(f"Train labels shape = {train_loader.dataset.tensors[1].shape}")


#%% create the model
class emnistNet(nn.Module):
    def __init__(self, print_toggle):
        super().__init__()
        
        self.print_toggle = print_toggle
        
        self.conv_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bnorm_1 = nn.BatchNorm2d(64) # channel input
        
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bnorm_2 = nn.BatchNorm2d(128)
        
        self.conv_3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bnorm_3 = nn.BatchNorm2d(256)
        
        self.conv_dropout = nn.Dropout(p=0.25)
        self.fc_dropout = nn.Dropout(p=0.5)
        
        self.fc_1 = nn.Linear(3*3*256, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 26)
        
    def forward(self, x):
        
        if self.print_toggle: print(f"Input: {list(x.shape)}")
        
        x = F.max_pool2d(self.conv_1(x), 2)
        x = F.leaky_relu(self.bnorm_1(x))
        x = self.conv_dropout(x)
        
        if self.print_toggle: print(f"First CPR block: {list(x.shape)}")
        
        x = F.max_pool2d(self.conv_2(x), 2)
        x = F.leaky_relu(self.bnorm_2(x))
        x = self.conv_dropout(x)
        
        if self.print_toggle: print(f"Second CPR block: {list(x.shape)}")
        
        x = F.max_pool2d(self.conv_3(x), 2)
        x = F.leaky_relu(self.bnorm_3(x))
        x = self.conv_dropout(x)
        
        if self.print_toggle: print(f"Third CPR block: {list(x.shape)}")

        n_units = x.shape.numel() // x.shape[0]
        x = x.view(-1, n_units)
        
        if self.print_toggle: print(f"Vectorized: {list(x.shape)}")
        
        
        x = F.leaky_relu(self.fc_1(x))
        x = self.fc_dropout(x)
        
        x = F.leaky_relu(self.fc_2(x))
        x = self.fc_dropout(x)
        
        x = self.fc_3(x)
        
        if self.print_toggle: print(f"Output: {list(x.shape)}")
        
        return x

def create_net(print_toggle=False):
    net = emnistNet(print_toggle)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    return net, criterion, optimizer


#%% test the model with one batch
net, criterion, optimizer = create_net(True)

X, y = next(iter(train_loader))
y_hat = net(X)

print(f"Output size: {y_hat.shape}")

loss = criterion(y_hat, torch.squeeze(y))
print(f"Loss: {loss.item()}")

print(summary(net, (1, 1, 28, 28)))

#%% Training
epochs = 10

def train_net():
    net, criterion, optimizer = create_net()
    
    net.to(device)
    
    train_loss, test_loss, train_err, test_err = (torch.zeros(epochs)
                                                  for _ in range(4))
    
    start_time = time.time()

    for epoch in range(epochs):
        
        batch_loss = []
        batch_err = []
        
        net.train()
        for x_train, y_train in train_loader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            y_train_pred = net(x_train)
            loss = criterion(y_train_pred, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
            batch_err.append(
                torch.mean((torch.argmax(
                        y_train_pred, axis=1) != y_train).float()).item())
            
        train_loss[epoch] = np.mean(batch_loss)
        train_err[epoch] = 100*np.mean(batch_err)
        
        net.eval()
        x_test, y_test = next(iter(test_loader))
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        with torch.inference_mode():
            y_test_pred = net(x_test)
            loss = criterion(y_test_pred, y_test)
            
        test_loss[epoch] = loss.item()
        test_err[epoch] = 100*torch.mean((torch.argmax(
                y_test_pred, axis=1) != y_test).float()).item()
        
        if epoch % 2 == 0:
            elapsed_time = time.time() - start_time
            print(f"time: {elapsed_time:.2f} |" \
                    f" epoch: {epoch}/{epochs} |" \
                      f" train loss: {train_loss[epoch]:.4f} |" \
                        f" test loss: {test_loss[epoch]:.4f} | " \
                          f" train error: {train_err[epoch]:.4f} |" \
                            f" test error: {test_err[epoch]:.4f} |")
        
    return train_loss, test_loss, train_err, test_err, net

train_loss, test_loss, train_err, test_err, net = train_net()
        

#%% Evaluation
# loss and error
fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(7,4), dpi=200)
ax_1.plot(train_loss, 's-', label="Train", markersize=4)
ax_1.plot(test_loss, 'o-', label="Test", markersize=4)
ax_1.set_xlabel("Epochs")
ax_1.set_ylabel("Loss (MSE)")

ax_2.plot(train_err, 's-', label="Train", markersize=4)
ax_2.plot(test_err, 'o-', label="Test", markersize=4)
ax_2.set_xlabel("Epochs")
ax_2.set_ylabel("Error (MSE)")
ax_2.set_title(f"Final test error: {test_err[-1]:.2f}%")

plt.suptitle("Train and Test loss and error")
plt.show()
        
# confusion matrix
x_test, y_test = next(iter(test_loader))
x_test, y_test = x_test.to(device), y_test.to(device)
y_pred = net(x_test)

c_matrix = skm.confusion_matrix(y_test.cpu(), torch.argmax(y_pred, axis=1).cpu(),
                                normalize='true')
fig = plt.figure(figsize=(10,10), dpi=200)
plt.imshow(c_matrix, 'Blues', vmax=0.05)

plt.xticks(range(26), labels=categories)
plt.yticks(range(26), labels=categories)   
plt.xlabel('Predicted letter')
plt.ylabel("True letter")
plt.title("Test confusion matrix")
plt.show()  
            