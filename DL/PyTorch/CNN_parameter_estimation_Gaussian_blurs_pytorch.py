import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from torchinfo import summary

from sklearn.model_selection import train_test_split

#%% Data
n_gauss = 2000
img_size = 91

def create_data():
    width = np.random.rand() * 10 + 5
    
    x = np.linspace(-4, 4, img_size)
    X,Y = np.meshgrid(x, x)
    
    images = torch.zeros(n_gauss, 1, img_size, img_size)
    labels = torch.zeros(n_gauss, 3)

    for i in range(n_gauss):
        width = np.random.rand() * 10 + 5
        loc = 1.5*np.random.randn(2)
        
        G = np.exp(-1 * ((X - loc[0])**2 + (Y + loc[1])**2) / width)
        G += np.random.randn(img_size, img_size) / 10
        
        images[i,:,:,:] = torch.Tensor(G).view(1, img_size, img_size)
        labels[i,:] = torch.Tensor([loc[0], loc[1], width])
        
    return images, labels

images, labels = create_data()

#%% Visualize
fig, axs = plt.subplots(3, 7, figsize=(13, 6), dpi=200, tight_layout=True)
for i, ax in enumerate(axs.flatten()):
    rdm_idx = np.random.randint(n_gauss)
    G = np.squeeze(images[rdm_idx])

    ax.imshow(G, vmin=-1, vmax=1, extent=[-4,4,-4,4], cmap='jet')
    
    ax.plot(labels[rdm_idx, 0], labels[rdm_idx, 1], 'bo', markersize=10)
    
    ax.set_title(f"X={labels[rdm_idx,0]:.0f}, Y={labels[rdm_idx,1]:.0f}," \
                 f" W={labels[rdm_idx,2]:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

#%% Create datasets
train_data, test_data, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2) 

train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0],
                         shuffle=False, drop_last=False)

train_samples = len(train_loader)*train_loader.batch_size
test_samples = len(test_loader)*test_loader.batch_size

print(f"train samples: {train_samples} | test samples: {test_samples}")

#%% Create model
class GausParamEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=4,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(22*22*4, 50),
            nn.ReLU(),
            nn.Linear(50, 3),
            )

    def forward(self, x):
        return self.enc(x)

def create_net():
    net = GausParamEncoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    return net, criterion, optimizer

net, criterion, optimizer = create_net()

#%% Test the model
print(summary(net, (batch_size, 1, img_size, img_size)))

#%% Model training
def train_net(images):
    epochs = 60
    
    net, criterion, optimizer = create_net()
    
    train_losses = torch.zeros(epochs)
    test_losses = torch.zeros(epochs)
    
    batch_loss = []
    print(f"Start training for {epochs} epochs")
    net.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            y_pred = net(X)
        
            loss = criterion(y_pred, y)
            batch_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_losses[epoch] = np.mean(batch_loss)
        
        X_test, y_test = next(iter(test_loader))
        net.eval()
        with torch.inference_mode():
            y_test_pred = net(X_test)
            loss = criterion(y_test_pred, y_test)
        
        test_losses[epoch] = loss.item()
        
        if epoch % 4 == 0:
            print(f"epoch: {epoch:4}/{epochs} " \
                  f"| train loss: {train_losses[epoch]:.4f} " \
                  f"| test loss: {test_losses[epoch]:.4f}")
    
    return train_losses, test_losses, net

#%% train the model
train_losses, test_losses, net = train_net(images)

#%% Model evaluation
# loss
fig = plt.figure(dpi=200)
plt.plot(train_losses, 's-', label="Train", markersize=4)
plt.plot(test_losses, 'o-', label="Test", markersize=4)
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Train and Test loss")
plt.show()

# Pass test set
X_test, y_test = next(iter(test_loader))
net.eval()
with torch.inference_mode():
    y_pred_test = net(X_test)

# plot predicted gaussian values on true gaussians
fig, axs = plt.subplots(2, 10, figsize=(16,4), dpi=200, tight_layout=True)

th = np.linspace(0, 2*np.pi)

for i, ax in enumerate(axs.flatten()):
    G = torch.squeeze(X_test[i,0,:,:]).detach()
    ax.imshow(G, vmin=-1, vmax=1, cmap='jet', extent=[-4,4,-4,4])
    ax.plot([-4,4], [0,0], 'w--')
    ax.plot([0,0], [-4,4], 'w--')
    
    c_x = y_pred_test[i][0].item()
    c_y = y_pred_test[i][1].item()
    rd = y_pred_test[i][2].item()
    
    x = c_x + np.cos(th)*np.sqrt(rd)
    y = c_y + np.sin(th)*np.sqrt(rd)
    
    ax.plot(x, y, 'b')
    ax.plot(c_x, c_y, 'bo')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
plt.show()

# error plots true vs predictions
fig, axs = plt.subplots(1, 3, figsize=(10,5), dpi=200, tight_layout=True)

title_labels = ['x', 'y', 'r']
for i, ax in enumerate(axs.flatten()):
    ax.plot(y_test[:, i], y_pred_test[:, i].detach(), 'o', alpha=0.6)
    ax.set_ylim([y_test[:, i].min()-0.5, y_test[:, i].max()+0.5])

    ax.set_title(title_labels[i])
    ax.grid()
    
axs[1].set_xlabel('True values')
axs[0].set_ylabel('Predicted values')

plt.show()
    