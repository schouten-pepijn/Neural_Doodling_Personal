mport torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from imageio import imread
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

plot_bool = True
mj_img = True
device = "cuda" if torch.cuda.is_available() else "cpu"

vgg_net = torchvision.models.vgg19(weights='IMAGENET1K_V1')

for p in vgg_net.parameters():
    p.requires_grad = False

vgg_net.eval()

if mj_img:
    # img_style = Image.open("/content/style_mj.jpeg")
    img_style = Image.open("/content/style_mj3.jpeg")
    img_content = Image.open("/content/content_pep1.jpeg")
    # img_content = Image.open("/content/content_mj1.jpeg")
    # img_content = Image.open("/content/content_mj2.jpeg")
    img_style = np.asarray(img_style)
    img_content = np.asarray(img_content)
else:
    img_content = imread("https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/De_nieuwe_vleugel_van_het_Stedelijk_Museum_Amsterdam.jpg/1920px-De_nieuwe_vleugel_van_het_Stedelijk_Museum_Amsterdam.jpg")
    img_style = imread("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg/220px-Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg")

img_target = np.random.randint(low=0, high=255, size=img_content.shape, dtype=np.uint8)

print(img_content.shape, img_style.shape, img_target.shape)

Transforms = T.Compose([
    T.ToTensor(),
    T.Resize(256),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if plot_bool:
    fig, ax = plt.subplots(1, 3, dpi=200, tight_layout=True)
    for i, img in enumerate((img_content, img_target, img_style)):
        ax[i].imshow(img)
        ax[i].axis('off')
    ax[0].set_title('Content', fontsize=8)
    ax[1].set_title('Target', fontsize=8)
    ax[2].set_title('Style', fontsize=8)
    plt.show()

img_content = Transforms(img_content).unsqueeze(0)
img_style = Transforms(img_style).unsqueeze(0)
img_target = Transforms(img_target).unsqueeze(0)

print(img_content.shape, img_style.shape, img_target.shape)

#%% functions
def getFeatureMapActs(img, net):
    feat_maps = []
    feat_names = []

    layer_idx = 0

    for layer in net.features:
        img = layer(img)
        if 'Conv2d' in str(layer):
            feat_maps.append(img)
            feat_names.append("ConvLayer_" + str(layer_idx))
            layer_idx += 1
    return feat_maps, feat_names


def gram_mat(M):
    _, channels, height, width = M.shape
    M = M.reshape(channels, height * width)
    gram = torch.mm(M, M.t()) / (channels * height * width)
    return gram


# some info
feat_maps, feat_names = getFeatureMapActs(img_content, vgg_net)

for i in range(len(feat_names)):
    print(f"Feature map {feat_names[i]} is size {feat_maps[i].shape}")

content_feat_maps, content_feat_names = getFeatureMapActs(img_content, vgg_net)
style_feat_maps, style_feat_names = getFeatureMapActs(img_style, vgg_net)

# show features maps and gam matrices of content and style maps
if plot_bool:
    # content maps
    fig, ax = plt.subplots(2, 5, figsize=(18,6), dpi=200, tight_layout=True)
    for i in range(5):
        pic = np.mean(content_feat_maps[i].squeeze().numpy(), axis=0)
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
        ax[0, i].imshow(pic, cmap='gray')
        ax[0, i].set_title("Content layer " + str(content_feat_names[i]), fontsize=8)

        pic = gram_mat(content_feat_maps[i]).numpy()
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
        ax[1, i].imshow(pic, cmap="gray")
        ax[1, i].set_title("Gram matrix, layer " + str(content_feat_names[i]), fontsize=8)
    plt.show()
    # style maps
    fig, ax = plt.subplots(2, 5, figsize=(18, 6), dpi=200, tight_layout=True)
    for i in range(5):
        pic = np.mean(style_feat_maps[i].squeeze().numpy(), axis=0)
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
        ax[0, i].imshow(pic, cmap='gray')
        ax[0, i].set_title("Content layer " + str(style_feat_names[i]), fontsize=8)

        pic = gram_mat(style_feat_maps[i]).numpy()
        pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
        ax[1, i].imshow(pic, cmap="gray")
        ax[1, i].set_title("Gram matrix, layer " + str(style_feat_names[i]), fontsize=8)
    plt.show()

#%% Style transfer
content_layers = [
    "ConvLayer_0",
    "ConvLayer_1",
   # "ConvLayer_2",
    "ConvLayer_4"
    ]
style_layers = [
    "ConvLayer_1",
    "ConvLayer_2",
    "ConvLayer_3",
    "ConvLayer_4",
    "ConvLayer_5"
    ]
style_layer_weights = [1, 0.5, 0.5, 0.2, 0.1]

style_scaling = 1e5
# style_scaling = 1e6

n_epochs = 1500

vgg_net.to(device)

img_content = img_content.to(device)
img_style = img_style.to(device)
target = img_target.clone().to(device)
target.requires_grad = True

optimizer = torch.optim.RMSprop([target], lr=5e-3)

content_feat_maps, content_feat_names = getFeatureMapActs(img_content, vgg_net)
style_feat_maps, style_feat_names = getFeatureMapActs(img_style, vgg_net)

for epoch in range(n_epochs):
    target_feat_maps, target_feat_names = getFeatureMapActs(target, vgg_net)
    style_loss = 0
    content_loss = 0

    for layer_i in range(len(target_feat_maps)):

        if target_feat_names[layer_i] in content_layers:
            content_loss += torch.mean((target_feat_maps[layer_i] - content_feat_maps[layer_i]) ** 2)

        if target_feat_names[layer_i] in style_layers:
            G_target = gram_mat(target_feat_maps[layer_i])
            G_style = gram_mat(style_feat_maps[layer_i])

            style_loss += torch.mean((G_target - G_style) ** 2) * style_layer_weights[
                style_layers.index(target_feat_names[layer_i])]

    loss = content_loss + style_scaling * style_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4} loss: {loss.item():.4f}")


#%% Show the result
fig, ax = plt.subplots(1, 3, figsize=(18, 11))
pic = img_content.squeeze().numpy().transpose(1, 2, 0)
pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
ax[0].imshow(pic)
ax[0].set_title("Content", fontsize=8)
ax[0].set_xticks([])
ax[0].set_yticks([])

pic = torch.sigmoid(target).detach().squeeze().numpy().transpose(1, 2, 0)
ax[1].imshow(pic)
ax[1].set_title('Target', fontsize=8)
ax[1].set_xticks([])
ax[1].set_yticks([])

pic = img_style.squeeze().numpy().transpose(1, 2, 0)
pic = (pic - np.min(pic)) / (np.max(pic) - np.min(pic))
ax[2].imshow(pic)
ax[2].set_title('Style', fontsize=8)
ax[2].set_xticks([])
ax[2].set_yticks([])

plt.show()
