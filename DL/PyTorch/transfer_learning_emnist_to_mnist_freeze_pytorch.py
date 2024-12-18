import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import copy
import time

from sklearn.model_selection import train_test_split

import torchvision

import matplotlib.pyplot as plt

print(torch.__version__)
print(torchvision.__version__)


#%% VARIABLES
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
test_batch_size = 256


#%% LETTER DATA
root = "/home/pepijn/Desktop/EMINST"
letter_data = torchvision.datasets.EMNIST(root, split='letters', download=True)

letter_categories = letter_data.classes[1:]
letter_labels = copy.deepcopy(letter_data.targets) -1

letter_images = letter_data.data.view([letter_data.data.shape[0],1,28,28]).float()
letter_images /= torch.max(letter_images)

train_data, test_data, train_labels, test_labels = train_test_split(
    letter_images, letter_labels, test_size=0.1)

train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

letter_train_loader = DataLoader(train_data, batch_size,
                                 shuffle=True, drop_last=True)
letter_test_loader = DataLoader(test_data, test_batch_size)


#%% NUMBER DATA
number_data = torchvision.datasets.EMNIST(root, split='digits', download=True)

number_images = number_data.data.view([number_data.data.shape[0],1,28,28]).float()
number_images /= torch.max(number_images)

number_labels = number_data.targets

train_data, test_data, train_labels, test_labels = train_test_split(
    number_images, number_labels, test_size=0.1)

train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

number_train_loader = DataLoader(train_data, batch_size,
                                 shuffle=True, drop_last=True)
number_test_loader = DataLoader(test_data, test_batch_size)


#%% MODEL
class emnistNet(nn.Module):
    def __init__(self, print_toggle):
        super().__init__()
        
        self.print_toggle = print_toggle
        
        self.conv_1 = nn.Conv2d(1, 6, 3, padding=1)
        self.bnorm_1 = nn.BatchNorm2d(6) # channel input
        
        self.conv_2 = nn.Conv2d(6, 6, 3, padding=1)
        self.bnorm_2 = nn.BatchNorm2d(6)
        
        self.fc_1 = nn.Linear(7*7*6, 50)
        self.fc_2 = nn.Linear(50, 26)
        
    def forward(self, x):
        
        if self.print_toggle: print(f"Input: {list(x.shape)}")
        
        x = F.max_pool2d(self.conv_1(x), 2)
        x = F.leaky_relu(self.bnorm_1(x))
        
        if self.print_toggle: print(f"First CPR block: {list(x.shape)}")
        
        x = F.max_pool2d(self.conv_2(x), 2)
        x = F.leaky_relu(self.bnorm_2(x))

        n_units = x.shape.numel() // x.shape[0]
        x = x.view(-1, n_units)
        
        if self.print_toggle: print(f"Vectorized: {list(x.shape)}")
        
        
        x = F.leaky_relu(self.fc_1(x))
        x = self.fc_2(x)
        
        if self.print_toggle: print(f"Output: {list(x.shape)}")
        
        return x

def create_net(print_toggle=False):
    net = emnistNet(print_toggle)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    return net, criterion, optimizer


#%% SOURCE MODEL
def train_model(net_metaparams, train_loader, test_loader, epochs=10):
    
    net, criterion, optimizer = net_metaparams
    
    train_losses = torch.zeros(epochs)
    test_losses = torch.zeros(epochs)
    train_acc, test_acc = [], []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        net.train()
        batch_train_acc, batch_train_loss = [], []
        batch_test_acc, batch_test_loss = [], []
        
        for x_train, y_train in train_loader:
            y_train_pred = net(x_train)
            
            loss = criterion(y_train_pred, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_train_acc.append(
                torch.mean(
                    (torch.argmax(
                        y_train_pred, axis=1) == y_train).float())*100)
            batch_train_loss.append(loss.item())
            
        train_acc.append(np.mean(batch_train_acc))
        train_losses[epoch] = np.mean(batch_train_loss)
        
        net.eval()
        for x_test, y_test in test_loader:
            with torch.inference_mode():
                y_test_pred = net(x_test)
            
            loss = criterion(y_test_pred, y_test)
            
            batch_test_acc.append(
                torch.mean(
                    (torch.argmax(
                        y_test_pred, axis=1) == y_test).float())*100)
            batch_test_loss.append(loss.item())
        
        test_acc.append(np.mean(batch_test_acc))
        test_losses[epoch] = np.mean(batch_test_loss)
        
        if epoch % 2 == 0:
            elapsed_time = time.time() - start_time
            print(f"time: {elapsed_time:.2f} |" \
                    f" epoch: {epoch}/{epochs} |" \
                      f" train loss: {train_losses[epoch]:.4f} |" \
                        f" test loss: {test_losses[epoch]:.4f} | " \
                          f" train accuracy: {train_acc[epoch]:.4f} |" \
                            f" test accuracy: {test_acc[epoch]:.4f} |")
        
    return train_acc, test_acc, train_losses, test_losses, net


def test_model(net, criterion, data_loader):

    batch_acc = []
    batch_loss = []
        
    net.eval()
    for x_test, y_test in data_loader:
        with torch.inference_mode():
            y_pred = net(x_test)
        
        loss = criterion(y_pred, y_test)
        
        batch_acc.append(
            torch.mean(
                (torch.argmax(
                    y_pred, axis=1) == y_test).float())*100)
        batch_loss.append(loss.item())
    
    accuracy = np.mean(batch_acc)
    losses = np.mean(batch_loss)
        
    return accuracy, losses


#%% TRAIN SOURCE MODEL
letter_net, criterion, optimizer = create_net(False)

train_acc, test_acc, train_losses, test_losses, letter_net = train_model(
    (letter_net, criterion, optimizer), letter_train_loader,
    letter_test_loader, epochs=10)   


#%% EVALUATE MODEL
# loss and error
fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(7,4),
                                 tight_layout=True, dpi=200)
ax_1.plot(train_losses, 's-', label="Train", markersize=4)
ax_1.plot(test_losses, 'o-', label="Test", markersize=4)
ax_1.set_xlabel("Epochs")
ax_1.set_ylabel("Loss (MSE)")

ax_2.plot(train_acc, 's-', label="Train", markersize=4)
ax_2.plot(test_acc, 'o-', label="Test", markersize=4)
ax_2.set_xlabel("Epochs")
ax_2.set_ylabel("Error (MSE)")
ax_2.set_title(f"Final test accuracy: {test_acc[-1]:.2f}%")

plt.suptitle("Train and Test loss and accuracy")
plt.show()  


#%% TEST ON NUMBER DATA
number_acc, number_loss = test_model(letter_net, criterion, number_test_loader)
print(f"LETTER accuracy performance on NUMBERS: {number_acc:.2f}%")


#%% FINE TUNE WITH ONE TRAINING EPOCH ON THE CLASSIFIER LAYER
number_net, criterion, optimizer = create_net(False)

# transfer pretrained weights to net model
for target, source in zip(letter_net.named_parameters(),
                          number_net.named_parameters()):
    target[1].data = copy.deepcopy(source[1].data)
 
# change classifier layer to output 10 features
print(number_net.fc_2)
number_net.fc_2 = nn.Linear(50, 10)
print(number_net.fc_2)

# freeze conv and batch norm layers
trainable_parameters = sum(p.numel() for p in number_net.parameters()
                           if p.requires_grad)
print(trainable_parameters)

for p in number_net.named_parameters():
    if "conv" in p[0]:
        p[1].requires_grad = False
    if "bnorm" in p[0]:
        p[1].requires_grad = False

# Freeze first linear layer (optional)
# number_net.fc_1.requires_grad = False

trainable_parameters = sum(p.numel() for p in number_net.parameters()
                           if p.requires_grad)
print(trainable_parameters)


#%% RETRAIN
train_acc, test_acc, train_losses, test_losses, fmnist_net = train_model(
    (number_net, criterion, optimizer), number_train_loader,
    number_test_loader, epochs=1)

#%% EVALUATE WITH NUMBERS
print(f"LETTER train accuracy on NUMBERS: {train_acc[-1]:.2f}%") 
print(f"LETTER test accuracy on NUMBERS: {test_acc[-1]:.2f}%") 
