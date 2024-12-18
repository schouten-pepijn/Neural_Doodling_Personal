import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#%% Import the data
path = 'data/mnist_train.csv'
full_data = np.loadtxt(open(path, 'rb'), delimiter=',', skiprows=1)

labels = full_data[:8, 0]
data = full_data[:8, 1:]

data_norm = data / np.max(data)
data_norm = data_norm.reshape(data_norm.shape[0], 1, 28, 28)

print(data_norm.shape)
print(labels.shape)

data_T  = torch.tensor(data_norm).float()
labels_T = torch.tensor(labels).float()

#%% Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, tensors, transform=None):

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors
        self.transform = transform

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, idx):
        if self.transform:
            x = self.transform(self.tensors[0][idx])
        else:
            x = self.tensors[0][idx]

        y = self.tensors[1][idx]

        return x, y


img_trans = T.Compose([
    T.ToPILImage(),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(90),
    T.ToTensor()
])

train_data = CustomDataset((data_T, labels_T), transform=img_trans)

train_dataloader = DataLoader(train_data, batch_size=8, shuffle=False)

#%% Evaluate
X, y = next(iter(train_dataloader))


fig, axs = plt.subplots(2, 8, figsize=(20, 4))

for i in range(8):
    axs[0,i].imshow(data_T[i,0,:,:].detach(), cmap='gray')
    axs[1,i].imshow(X[i,0,:,:].detach(), cmap='gray')

    for row in range(2):
        axs[row,i].set_xticks([])
        axs[row,i].set_yticks([])

axs[0,0].set_ylabel('Original')
axs[1,0].set_ylabel('Custom dataset')

plt.show()

