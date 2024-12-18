import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
 
import numpy as np
import matplotlib.pyplot as plt

from torchinfo import summary

from sklearn.model_selection import train_test_split


#%% Data
n_class = 1000
img_size = 91

x = np.linspace(-4, 4, img_size)
X,Y = np.meshgrid(x, x)

widths = [1.8, 2.4]

images = torch.zeros(2*n_class, 1, img_size, img_size)
labels = torch.zeros(2*n_class)

for i in range(2*n_class):
    ro = 2*np.random.randn(2)
    G = np.exp(-1*((X - ro[0])**2 + (Y - ro[1])**2) / (2*widths[i%2]**2))
    
    G += np.random.randn(img_size, img_size) / 5
    
    images[i] = torch.Tensor(G).view(1, img_size, img_size)
    labels[i] = i%2

labels = labels[:, np.newaxis]

#%% Visualize
fig, axs = plt.subplots(3, 7, figsize=(13, 6), dpi=200, tight_layout=True)
for i, ax in enumerate(axs.flatten()):
    rdm_idx = np.random.randint(2 * n_class)
    G = np.squeeze(images[rdm_idx])
    ax.imshow(G, vmin=-1, vmax=1, cmap='jet')
    ax.set_title(f"Class {int(labels[rdm_idx].item())}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
        
#%% Train / test dataloader
train_data, test_data, train_labels, test_labels = train_test_split(images,
                                                                    labels,
                                                                    test_size=0.2)

train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                          drop_last=True)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0],
                         shuffle=False, drop_last=False)

print(train_loader.dataset.tensors[0].shape,
      train_loader.dataset.tensors[1].shape)

#%% Create model
class GausNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=4, kernel_size=3, padding=1)
  
        self.fc1 = nn.Linear(in_features=22*22*4, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=1)
            

    def forward(self, x):
        conv1_act = F.relu(self.conv1(x))
        x = F.avg_pool2d(conv1_act, (2, 2))
        
        conv2_act = F.relu(self.conv2(x))
        x = F.avg_pool2d(conv2_act, (2, 2))
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, conv1_act, conv2_act


def create_net():
    net = GausNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    return net, criterion, optimizer
    
#%% Test the network
net, criterion, optimizer = create_net()

X, y = next(iter(train_loader))
y_hat, feat_map1, feat_map2 = net(X)
loss = criterion(y_hat, y)    

print("  ", y_hat.shape)
print("  ", f"Loss: {loss}")

summary(net, (1, 1, img_size, img_size))

#%% Training
n_epochs = 100

train_loss, test_loss, train_acc, test_acc = (
    torch.zeros(n_epochs) for _ in range(4))

for epoch in range(n_epochs):
    batch_loss, batch_acc = [], []
    net.train()
    for X_train, y_train in train_loader:
        
        y_hat_train = net(X_train)[0]
        loss = criterion(y_hat_train, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())
        batch_acc.append(torch.mean(((y_hat_train > 0) == y_train).float()).item())
        
    train_loss[epoch] = np.mean(batch_loss)
    train_acc[epoch] = 100 * np.mean(batch_acc)
    
    X_test, y_test = next(iter(test_loader))
    net.eval()
    with torch.inference_mode():
        y_hat_test, feat_map1, feat_map2  = net(X_test)
        loss = criterion(y_hat_test, y_test)
    
    test_loss[epoch] = loss.item()
    test_acc[epoch] = 100 * torch.mean(((y_hat_test > 0) == y_test).float()).item()
    
    
    if epoch % 10 == 0:
        print(f"epoch: {epoch:4}/{n_epochs} "
              f"| train loss: {train_loss[epoch]:3f} " \
                  f"| train_acc: {train_acc[epoch]:2f} " \
                      f"| test_loss: {test_loss[epoch]:3f} " \
                          f"| test_acc: {test_acc[epoch]:2f}")
        
    
#%% Kernel exploration
layer_1w = net.conv1.weight.squeeze().detach().numpy()
layer_3w = net.conv2.weight.squeeze().detach().numpy()
fig, axs = plt.subplots(5, 6, figsize=(15, 3), dpi=200)
for i, ax in enumerate(axs.flatten()):
    if i < 6:
        ax.imshow(layer_1w[i], cmap="Purples")
        ax.axis('off')
    else:
        i -= 6
        idx = np.unravel_index(i, (4, 6))
        ax.imshow(layer_3w[idx[0], idx[1]], cmap="Purples")
        ax.axis('off')
plt.suptitle("Convolutional kernels")
plt.show()

#%% Feature maps exploration
# first convolutional layer
fig, axs = plt.subplots(7, 10, figsize=(12, 6), tight_layout=True)
for pic_i in range(10):
    img = X_test[pic_i, 0, :, :].detach()
    axs[0, pic_i].imshow(img, cmap='jet', vmin=0, vmax=1)
    axs[0, pic_i].axis('off')
    
    for feat_i in range(6):
        img = feat_map1[pic_i, feat_i, :, :].detach()
        axs[feat_i + 1, pic_i].imshow(img, cmap='inferno', vmin=0, vmax=torch.max(img)*0.9)
        axs[feat_i + 1, pic_i].axis('off')
plt.suptitle("Feature maps of first conv activations")
plt.show()
      
# second convolutional layer 
fig, axs = plt.subplots(5, 10, figsize=(12, 6), tight_layout=True)
for pic_i in range(10):
    img = X_test[pic_i, 0, :, :].detach()
    axs[0, pic_i].imshow(img, cmap='jet', vmin=0, vmax=1)
    axs[0, pic_i].axis('off')
    
    for feat_i in range(4):
        img = feat_map2[pic_i, feat_i, :, :].detach()
        axs[feat_i + 1, pic_i].imshow(img, cmap='inferno', vmin=0, vmax=torch.max(img)*0.9)
        axs[feat_i + 1, pic_i].axis('off')
plt.suptitle("Feature maps of second conv activations")
plt.show()

#%% spatial correlation maps between feature maps of second convolutional layer
n_stims, n_maps = feat_map2.shape[0], feat_map2.shape[1]
n_cors = (n_maps * (n_maps - 1)) // 2

all_rs = np.zeros((n_stims, n_cors))
C_all = np.zeros((n_maps, n_maps))

for i in range(n_stims):
    flat_featmap = feat_map2[i, :, :, :].reshape(n_maps, -1).detach()
    
    C = np.corrcoef(flat_featmap)
    C_all += C
    
    idx = np.nonzero(np.triu(C, 1))
    all_rs[i, :] = C[idx]

x_lab = []*n_cors
for i in range(n_cors):
    x_lab.append(f"{idx[0][i]}-{idx[1][i]}")

fig = plt.figure(figsize=(16, 5))
ax0 = fig.add_axes([0.1, 0.1, 0.55, 0.9])
ax1 = fig.add_axes([0.68, 0.1, 0.3, 0.9])
cax = fig.add_axes([0.98, 0.1, 0.01, 0.9])

for i in range(n_cors):
    ax0.plot(i+np.random.randn(n_stims) / 30, all_rs[:, i] , 'o',
             markerfacecolor='w', markersize=10)

ax0.set_xlim([-0.5, n_cors - 0.5])
ax0.set_ylim([-1.05, 1.05])
ax0.set_xticks(range(n_cors))
ax0.set_xticklabels(x_lab)
ax0.set_xlabel("Feature map pair")
ax0.set_ylabel("Correlation coefficient")
ax0.set_title("Correlations for each image")

h = ax1.imshow(C_all / n_stims, vmin=-1, vmax=1)
ax1.set_title("Correlation matrix")
ax1.set_xlabel("Feature map")
ax1.set_ylabel("Feature map")
fig.colorbar(h, cax=cax)
plt.show()
    