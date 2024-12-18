import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

#%% DATA LOAD
directory = Path("/run/media/pepijn/37ac6286-af0a-4bfb-890f-09f2e9eca83e" \
                 "/Datasets/MNIST_IMG/")
    
transforms = torchvision.transforms.Compose({
    torchvision.transforms.ToTensor()})
    
train_data = torchvision.datasets.MNIST(root=directory,
                                        train=True,
                                        download=True,
                                        transform=transforms).data

train_labels = torchvision.datasets.MNIST(root=directory,
                                        train=True,
                                        download=True,
                                        transform=transforms).targets

test_data = torchvision.datasets.MNIST(root=directory,
                                       train=False,
                                       download=True,
                                       transform=transforms).data

test_labels = torchvision.datasets.MNIST(root=directory,
                                        train=False,
                                        download=True,
                                        transform=transforms).targets

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
            
            self.input = nn.Linear(in_shape, 150)
            self.enc = nn.Linear(150, 15)
            self.lat = nn.Linear(15, 150)
            self.dec = nn.Linear(150, in_shape)
            
        def forward(self, x):
            x = F.relu(self.input(x))
            latent = F.relu(self.enc(x))
            x = F.relu(self.lat(latent))
            x = torch.sigmoid(self.dec(x))
            return x, latent
            
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
        
        yHat, latent = net(X)
        loss = criterion(yHat, X)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses[epoch] = loss.item()
        
        if epoch % 100 == 0:
            print(f'Epoch: {epoch:5} | Loss: {losses[epoch]:.5f}')
        
    return losses, net, latent
        
losses, net, latent = trainModel(scaled_train_data)

#%% Loss evaluation
print(f'Final loss: {losses[-1]:.4f}')
plt.figure(dpi=200)
plt.plot(losses, '-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Final loss: {losses[-1]:.4f}')
plt.show()

#%% test data
with torch.inference_mode():
    yHat_test, latent_test = net(scaled_test_data)

# latent states
fig, axs = plt.subplots(1, 2, dpi=200, tight_layout=True)
axs[0].hist(latent_test.ravel(), 100)
axs[0].set_xlabel('Latent activation value')
axs[0].set_ylabel('Count')
axs[0].set_title('Distribution of latent units activations')
img = axs[1].imshow(latent_test, aspect='auto', vmin=0, vmax=10)
axs[1].set_xlabel('Latent node')
axs[1].set_ylabel('Image number')
axs[1].set_title('All latent activations')
plt.colorbar(img, ax=axs[1])
plt.show()

#%% Principal Component Analysis
# variance explained per PC
pca_data = PCA(n_components=15).fit(scaled_test_data)
pca_latent = PCA().fit(latent_test)

plt.figure(dpi=200)
plt.plot(100*pca_data.explained_variance_ratio_, 's-', label='Data PCA')
plt.plot(100*pca_latent.explained_variance_ratio_, 'o-', label='Latent PCA')
plt.xlabel('Components')
plt.ylabel('Percent variance explained')
plt.title('Variance explained by PC in data and latent')
plt.legend()
plt.show()

# eigenvector plots
scores_data = pca_data.fit_transform(scaled_test_data)
scores_latent = pca_latent.fit_transform(latent_test)

fig, axs = plt.subplots(1, 3, dpi=200, tight_layout=True)
data_scatter = axs[0].scatter(scores_data[:, 0], scores_data[:, 1],
                              c=test_labels, s=3, alpha=0.4, cmap='tab10')
latent_scatter = axs[1].scatter(latent_test[:, 0], latent_test[:, 1],
                                c=test_labels, s=3, alpha=0.4, cmap='tab10')
axs[0].set_title('Data PC1 vs 2')
axs[1].set_title('Latent PC1 vs 2')
axs[2].legend(*data_scatter.legend_elements())
axs[2].set_xticks([])
axs[2].set_yticks([])
plt.show()

fig, axs = plt.subplots(1, 3, dpi=200,
                        sharey=True, tight_layout=True)
data_scatter = axs[0].scatter(scores_data[:, 0], scores_data[:, 1],
                              c=test_labels, s=3, alpha=0.4, cmap='tab10')
latent_scatter = axs[1].scatter(latent_test[:, 0], latent_test[:, 1],
                                c=test_labels, s=3, alpha=0.4, cmap='tab10')
axs[0].set_title('Data PC1 vs 2')
axs[1].set_title('Latent PC1 vs 2')
axs[2].legend(*data_scatter.legend_elements())
axs[2].set_xticks([])
plt.show()