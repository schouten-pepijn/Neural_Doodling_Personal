import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T

from torchinfo import summary

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import copy
import time
import sys

#%% VARIABLES
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 100

#%% DATA
root = "/home/pepijn/Desktop/MNIST"
data = torchvision.datasets.MNIST(root, train=True,
                                  download=True).data

data_shape = data.shape

scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data.view((data.shape[0], -1)))
print(f"data min: {data.min()}, data max: {data.max()}")
data_T = torch.tensor(data).float()


#%% MODEL
class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)
        
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.out(x)
        return self.sigmoid(x)


class generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 28*28)
        
        self.lrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.out(x)
        return self.tanh(x)


dnet = discriminator()
y_sample = dnet(torch.randn(10, 784))
print(y_sample)

gnet = generator()
y_sample = gnet(torch.randn(10, 64))

plt.figure()
plt.imshow(y_sample[0].detach().squeeze().view(28, 28))
plt.show()
    
        

#%% TRAINING
def train_net(epochs=50000):
    criterion = nn.BCELoss()
    
    dnet = discriminator().to(device)
    gnet = generator().to(device)
    
    d_optimizer = torch.optim.Adam(dnet.parameters(), lr=3e-4)
    g_optimizer = torch.optim.Adam(gnet.parameters(), lr=3e-4)
    
    losses = np.zeros((epochs, 2))
    disc_decision = np.zeros((epochs, 2))
    
    for epoch in range(epochs):
        # mini batches of real and fake images
        rdm_idx = torch.randint(data_T.shape[0], (batch_size,))
        real_imgs = data_T[rdm_idx,:].to(device)
        fake_imgs = gnet(torch.randn(batch_size, 64).to(device))
        
        # labels for real and fake images
        real_labs = torch.ones(batch_size, 1).to(device)
        fake_labs = torch.zeros(batch_size, 1).to(device)
        
        # train discriminator
        # fwd pass read
        pred_real = dnet(real_imgs)
        d_loss_real = criterion(pred_real, real_labs)
        
        # fwd pass fake
        pred_fake = dnet(fake_imgs)
        d_loss_fake = criterion(pred_fake, fake_labs)
        
        # combined loss
        d_loss = d_loss_real + d_loss_fake
        
        losses[epoch, 0] = d_loss.item()
        disc_decision[epoch, 0] = torch.mean((pred_real>0.5).float()).detach()
        
        # backprop
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # train generator
        # generate fake imgs
        fake_imgs = gnet(torch.randn(batch_size, 64).to(device))
        pred_fake = dnet(fake_imgs)
        
        # compute loss
        g_loss = criterion(pred_fake, real_labs)
        
        losses[epoch, 1] = g_loss.item()
        disc_decision[epoch, 1] = torch.mean((pred_fake>0.5).float()).detach()
        
        # backprop
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (epoch+1) % 1000 == 0:
            msg = f"epoch: {epoch+1:5}/{epochs} | d_loss: {losses[epoch, 0]:.3f} | g_loss: {losses[epoch, 1]:.3f}"
            print(msg)
            
            plt.figure()
            plt.imshow(fake_imgs[0].reshape(28,28).cpu().detach().numpy(), cmap="gray")
            plt.title(f"Discriminator decision: {pred_fake[0].squeeze().cpu().detach().numpy()>0.5}")
            plt.axis('off')
            plt.show()
    return losses, disc_decision, dnet, gnet

losses, disc_decision, dnet, gnet = train_net()
    

#%% EVALUATION
fig, ax = plt.subplots(1, 3, figsize=(18, 5), dpi=200, tight_layout=True)
ax[0].plot(losses)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Model loss')
ax[0].legend(['Discriminator', 'Generator'])

ax[1].plot(losses[::5, 0], losses[::5, 1], 'k.', alpha=0.1)
ax[1].set_xlabel('Discriminator loss')
ax[1].set_ylabel('Generator loss')

ax[2].plot(disc_decision)
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Probability ("real")')
ax[2].set_title('Discriminator output')
ax[2].legend(['Real', 'Fake'])
plt.show()


#%% EVALUATE SDME GENERATED IMAGES
gnet.eval()
with torch.inference_mode():
    fake_data = gnet(torch.randn(12, 64).to(device)).cpu()

fig, axs = plt.subplots(3, 4, figsize=(8, 6), dpi=200, tight_layout=True)
for i, ax in enumerate(axs.flatten()):
    ax.imshow(fake_data[i,:,].detach().view(28, 28), cmap='gray')
    ax.axis('off')
plt.show()