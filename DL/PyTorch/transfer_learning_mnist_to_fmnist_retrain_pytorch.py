import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import copy
import time

#%% VARIABLES
batch_size = 32
test_batch_size = 128

#%% MNIST DATA
root_mnist = "/home/pepijn/Desktop/MNIST"
mnist_data = torchvision.datasets.MNIST(root_mnist, train=True, download=True)

mnist_images = mnist_data.data.unsqueeze(1).float()
mnist_images /= torch.max(mnist_images)
print(f"min/max mnist values: {torch.min(mnist_images)}/{torch.max(mnist_images)}")

mnist_labels = mnist_data.targets

(mnist_images_train, mnist_images_test,
 mnist_labels_train, mnist_labels_test) = train_test_split(
     mnist_images, mnist_labels, test_size=0.2)
     
mnist_train = TensorDataset(mnist_images_train, mnist_labels_train)
mnist_test = TensorDataset(mnist_images_test, mnist_labels_test)

mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size,
                                shuffle=True, drop_last=True)
mnist_test_loader = DataLoader(mnist_test, batch_size=test_batch_size,
                               shuffle=False, drop_last=False)

#%% FMNIST DATA
root_fmnist = "/home/pepijn/Desktop/FMNIST"
fmnist_data = torchvision.datasets.FashionMNIST(root_fmnist, train=True, download=True)

fmnist_images = fmnist_data.data.unsqueeze(1).float()
fmnist_images /= torch.max(fmnist_images)
print(f"min/max fmnist values: {torch.min(fmnist_images)}/{torch.max(fmnist_images)}")

fmnist_labels = fmnist_data.targets

(fmnist_images_train, fmnist_images_test,
 fmnist_labels_train, fmnist_labels_test) = train_test_split(
     fmnist_images, fmnist_labels, test_size=0.2)
     
fmnist_train = TensorDataset(fmnist_images_train, fmnist_labels_train)
fmnist_test = TensorDataset(fmnist_images_test, fmnist_labels_test)

fmnist_train_loader = DataLoader(fmnist_train, batch_size=batch_size,
                                 shuffle=True, drop_last=True)
fmnist_test_loader = DataLoader(fmnist_test, batch_size=test_batch_size,
                                shuffle=False, drop_last=False)

#%% MODEL
def createMNISTNet(print_toggle=False):
    class mnistNet(nn.Module):
        def __init__(self, print_toggle):
            super().__init__()

            self.conv1 =  nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=1)
            self.conv2 =  nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)

            self.fc1 = nn.Linear(180, 50) # 180
            self.out = nn.Linear(50, 10)

            self.print = print_toggle

        def forward(self, x):

            if self.print: print(f"Input: {x.shape}")

            x = F.relu(F.max_pool2d(self.conv1(x), 2))

            if self.print: print(f"Layer conv1/pool1: {x.shape}")

            x = F.relu(F.max_pool2d(self.conv2(x), 3))

            if self.print: print(f"Layer conv2/pool2: {x.shape}")

            n_units = x.shape.numel() // x.shape[0]
            x = x.view(-1, n_units)

            if self.print: print(f"Flattened view: {x.shape}")

            x = F.relu(self.fc1(x))

            if self.print: print(f"Layer fc1: {x.shape}")

            x = self.out(x)

            if self.print: print(f"Layer out: {x.shape}")

            return x

    net = mnistNet(print_toggle)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005)

    return net, criterion, optimizer

#%% TRAINNING
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
mnist_net, criterion, optimizer = createMNISTNet()


train_acc, test_acc, train_losses, test_losses, mnist_net = train_model(
    (mnist_net, criterion, optimizer), mnist_train_loader,
    mnist_test_loader, epochs=16)     

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
              
#%% EVALUATE WITH FMNIST
fmnist_acc, fmnist_loss = test_model(mnist_net, criterion, fmnist_test_loader)
print(f"MNISTNET performance on FMNIST: {fmnist_acc:.2f}%")

#%% FINE TUNE WITH ONE TRAINING ROUND
fmnist_net, criterion, optimizer = createMNISTNet()

# transfer pretrained weights to net model
for target, source in zip(fmnist_net.named_parameters(),
                          mnist_net.named_parameters()):
    target[1].data = copy.deepcopy(source[1].data)

#%% RETRAIN
train_acc, test_acc, train_losses, test_losses, fmnist_net = train_model(
    (fmnist_net, criterion, optimizer), fmnist_train_loader,
    fmnist_test_loader, epochs=1)  

#%% EVALUATE WITH FMNIST
print(f"MNISTNET train accuracy on FMNIST: {train_acc[-1]:.2f}%") 
print(f"MNISTNET test accuracy on FMNIST: {test_acc[-1]:.2f}%") 