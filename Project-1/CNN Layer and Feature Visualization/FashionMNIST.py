import torch
import torchvision

#data loading and transforming
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors for input into a CNN

#Definning transform to read the data in as a tensor
data_transform = transforms.ToTensor()

# choose the training and tes datasets
train_data = FashionMNIST(root='./data', train =True,
                              download=False, transform = data_transform)
print('Train data, number of images: ', len(train_data))

#Data iteration and batching
batch_size = 20
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#This cell iterates over the training dataset, 
#loading a random batch of image/label data, using dataiter.next(). 
#It then plots the batch of images and labels in a 2 x batch_size/2 grid.

import numpy as np
import matplotlib.pyplot as plt

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25,4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[],yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(classes[labels[idx]])
    
#select image by index
idx = 2
img = np.squeeze(images[idx])

#display the pixel values in that image
fig = plt.figure(figsize= (12,12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')