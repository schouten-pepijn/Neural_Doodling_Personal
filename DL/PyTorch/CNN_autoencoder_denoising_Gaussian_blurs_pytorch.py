import torch
import torch.nn as nn
 
import numpy as np
import matplotlib.pyplot as plt

from torchinfo import summary


#%% Data
n_gauss = 2000
img_size = 91

def create_data():
    widths = np.linspace(2, 20, n_gauss)
    
    x = np.linspace(-4, 4, img_size)
    X,Y = np.meshgrid(x, x)
    
    
    images_clean = torch.zeros(n_gauss, 1, img_size, img_size)
    images_occl = torch.zeros(n_gauss, 1, img_size, img_size)
    
    i_loc = np.arange(2, 28)
    i_width = np.arange(2, 5)
    
    for i in range(n_gauss):
        ro = 1.5*np.random.randn(2)
        G = np.exp(-1*((X - ro[0])**2 + (Y - ro[1])**2) / widths[i])
        
        G += np.random.randn(img_size, img_size) / 5
        
        images_clean[i, :, :, :] = torch.Tensor(G).view(1, img_size, img_size)
        
        i_1 = np.random.choice(i_loc)
        i_2 = np.random.choice(i_width)
    
        if np.random.randn() > 0:
            G[i_1:i_1+i_2, :] = 1
        else:
            G[:, i_1:i_1+i_2] = 1 
        
        images_occl[i, :, :, :] = torch.Tensor(G).view(1, img_size, img_size)
    
    return images_clean, images_occl

images_clean, images_occl = create_data()

#%% Visualize
fig, axs = plt.subplots(3, 7, figsize=(13, 6), dpi=200, tight_layout=True)
for i, ax in enumerate(axs.flatten()):
    rdm_idx = np.random.randint(n_gauss)
    if np.random.randint(2) < 1:
        G = np.squeeze(images_occl[rdm_idx])
    else:
        G = np.squeeze(images_clean[rdm_idx])
    ax.imshow(G, vmin=-1, vmax=1, cmap='jet')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

#%% Create model
class GaussAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=12, out_channels=6,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=6, out_channels=12,
                               kernel_size=3, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=12, out_channels=1,
                               kernel_size=3, stride=2)
            )
        
    def forward(self, x):
        return self.dec(self.enc(x))
    
def create_net():
    net = GaussAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    return net, criterion, optimizer

#%% Test the model
net, criterion, optimizer = create_net()

y_sample = net(images_occl[:10, :, :, :])

print(" ", y_sample.shape)

fig, ax = plt.subplots(1, 2, figsize=(8,3), dpi=200)
ax[0].imshow(torch.squeeze(images_occl[0, 0, :, :]).detach(), cmap='jet')
ax[0].set_title("Model input")

ax[1].imshow(torch.squeeze(y_sample[0, 0, :, :]).detach(), cmap='jet')
ax[1].set_title("Model output")
plt.show()

print(summary(net, (1, img_size, img_size)))

#%% Model training
def train_net(images, clean_image=None):
    epochs = 1000
    
    net, criterion, optimizer = create_net()
    
    losses = torch.zeros(epochs)
    
    print(f"Start training for {epochs} epochs")
    net.train()
    for epoch in range(epochs):
        batch = np.random.choice(n_gauss, size=32, replace=False)
        x_train = images[batch, :, :, :]
        y_train = net(x_train)
        
        if clean_image is not None:
            x_clean = clean_image[batch, :, :, :]
            loss = criterion(y_train, x_clean)
        else:
            loss = criterion(y_train, x_train)
        losses[epoch] = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"epoch: {epoch:4}/{epochs} | loss: {loss:.4f}")
    
    return losses, net

#%% train the model
losses, net = train_net(images_occl)

#%% Model evaluation
# loss
fig = plt.figure(dpi=200)
plt.plot(losses, 's-', label="Train")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title(f"Model loss (final loss={losses[-1]:.3f})")
plt.show()

# test samples
test_sample = np.random.choice(n_gauss, 300, replace=False)
x_test = images_occl[test_sample, :, :, :]
with torch.inference_mode():
    y_test = net(x_test)

# correlation coefficient
corr_all = np.zeros(len(test_sample))
for i, (x_sample, y_sample) in enumerate(zip(x_test, y_test)):
    x_sample = x_sample.squeeze().ravel()
    y_sample = y_sample.squeeze().ravel()
    
    corr = np.corrcoef(x_sample, y_sample)
    corr = corr[corr != 1].mean()
    corr_all[i] = corr

plt.figure(dpi=200)
plt.plot(np.random.normal(0, 0.01, len(test_sample)), corr_all,
         'o', markersize=4, markerfacecolor='none')
plt.title(f'Average correlation coef: {np.mean(corr_all):.2f}')
plt.show()

# some images
fig, axs = plt.subplots(2, 10, figsize=(18, 4),
                        dpi=200, tight_layout=True)
for i in range(10):
    G = x_test[i, :, :, :].squeeze().detach().numpy()
    O = y_test[i, :, :, :].squeeze().detach().numpy()
    
    axs[0, i].imshow(G, vmin=-1, vmax=1, cmap='jet')
    axs[0, i].axis('off')
    axs[1, i].imshow(O, vmin=-1, vmax=1, cmap='jet')
    axs[1, i].axis('off')
plt.suptitle("Model input versus model output")
plt.show()

#%% New model training
"""
train to remove bars
"""

losses, net = train_net(images_occl, images_clean)

#%% Model evaluation
# loss
fig = plt.figure(dpi=200)
plt.plot(losses, 's-', label="Train")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title(f"Model loss (final loss={losses[-1]:.3f})")
plt.show()

# test samples
test_sample = np.random.choice(n_gauss, 300, replace=False)
x_test = images_occl[test_sample, :, :, :]
with torch.inference_mode():
    y_test = net(x_test)

# some images
fig, axs = plt.subplots(2, 10, figsize=(18, 4),
                        dpi=200, tight_layout=True)
for i in range(10):
    G = x_test[i, :, :, :].squeeze().detach().numpy()
    O = y_test[i, :, :, :].squeeze().detach().numpy()
    
    axs[0, i].imshow(G, vmin=-1, vmax=1, cmap='jet')
    axs[0, i].axis('off')
    axs[1, i].imshow(O, vmin=-1, vmax=1, cmap='jet')
    axs[1, i].axis('off')
plt.suptitle("Model input versus model output")
plt.show()
