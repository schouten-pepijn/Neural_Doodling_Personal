import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import copy
import time

from torchinfo import summary

from sklearn.model_selection import train_test_split

import torchvision
import torchvision.transforms as T

import matplotlib.pyplot as plt

print(torch.__version__)
print(torchvision.__version__)


#%% VARIABLES
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
test_batch_size = 256
epochs = 10


#%% DATA STL10
root = "/home/pepijn/Desktop/STL10"

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])

train_set = torchvision.datasets.STL10(root, split='train', transform=transforms,
                                       download=True)
test_set = torchvision.datasets.STL10(root, split='test', transform=transforms,
                                      download=True)

train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, drop_last=True)
test_loader = DataLoader(test_set, batch_size=test_batch_size,
                         shuffle=False, drop_last=False)

print(f"Data shapes: {train_set.data.shape} / {test_set.data.shape}")
print(f"Data value range: {np.min(train_set.data)}, {np.max(train_set.data)}")
print(f"Data categories: {train_set.classes}")


#%% IMPORT RESNET18
weights = torchvision.models.ResNet18_Weights.DEFAULT
resnet = torchvision.models.resnet18(weights=weights)

print(resnet)
print(summary(resnet.to(device), (1, 3, 96, 96)))

# freeze layers
for p in resnet.parameters():
    p.requires_grad = False
    
# change classifying layer
resnet.fc = nn.Linear(512, 10)

resnet.to(device)


#%% TRAIN the model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=1e-3, momentum=0.9)

train_loss, train_acc = torch.zeros(epochs), torch.zeros(epochs)
for epoch in range(epochs):
    resnet.train()
    batch_loss, batch_acc = [], []
    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        y_pred = resnet(x_train)
        loss = criterion(y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())
        batch_acc.append(
            torch.mean((torch.argmax(y_pred, axis=1) == y_train).float()).item())
    
    train_loss[epoch] = np.mean(batch_loss)
    train_acc[epoch] = 100*np.mean(batch_acc)
    
    if epoch % 2 == 0:
        print(f" epoch: {epoch}/{epochs} |" \
                f" train loss: {train_loss[epoch]:.4f} |" \
                    f" train accuracy: {train_acc[epoch]:.4f} |")


#%% EVALUATE
resnet.eval()
batch_loss, batch_acc = [], []
for x_test, y_test in test_loader:
    x_test, y_test = x_test.to(device), y_test.to(device)

    with torch.inference_mode():
        y_pred = resnet(x_test)
    
    loss = criterion(y_pred, y_test)
    
    batch_loss.append(loss.item())
    batch_acc.append(
        torch.mean((torch.argmax(y_pred, axis=1) == y_test).float()).item())
    
test_loss = np.mean(batch_loss)
test_acc = 100*np.mean(batch_acc)


#%% VISUALIZE PERFORMANCE
x_sample, y_sample = next(iter(test_loader))
x_sample, y_sample = x_sample.to(device), y_sample.to(device)
resnet.eval()
preds = torch.argmax(resnet(x_sample), axis=1)

fig, axs = plt.subplots(4, 4, figsize=(10,10),
                        tight_layout=True, dpi=200)
for i, ax in enumerate(axs.flatten()):
    pic = x_sample.data[i].cpu().numpy().transpose((1, 2, 0))
    pic -= np.min(pic)
    pic /= np.max(pic)
    
    ax.imshow(pic)
    
    label = train_set.classes[preds[i]]
    true_class = train_set.classes[y_sample[i]]
    
    title = f"Pred: {label}  -  True: {true_class}"
    title_color = "g" if true_class == label else "r"
    ax.text(48, 90, title, ha="center", va="top", fontweight="bold",
            color="k", backgroundcolor=title_color, fontsize=8)
    ax.axis("off")
plt.show()

    
        
        
    
        