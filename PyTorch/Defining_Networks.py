import numpy as np
import torch, helper
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                                ])
trainset = datasets.MNIST('MNIST_data/', download=True,train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=True,train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');

from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()   #call init class of nn.Module
        
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        
        return x

model = Net()
print(model)

#print(model.fc1.weight)
#print(model.fc1.bias)

model.fc1.bias.data.fill_(0)
model.fc1.weight.data.normal_(std=0.1)
print(model.fc1.weight)
print(model.fc1.bias)

##forward pass
images, labels = next(iter(trainloader))

#Convert 28*28 image vector to 784*1 long vector
images.resize_(images.shape[0],1,784)   #images.shape[0] = batch size=64
ps = model.forward(images[0])

#Another way for building model
input_size = 748
hidden_sizes = [128,64]
output_size = 10

#Build a feedforward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Softmax(dim=1))
print(model)


##Naming layers
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_sizes[0])),
            ('relu1', nn.ReLU()),
            ('fc2',nn.Linear(hidden_sizes[0],hidden_sizes[1])),
            ('relu2',nn.ReLU()),
            ('fc3', nn.Linear(hidden_sizes[1],output_size)),
            ('Softmax', nn.Softmax(dim=1))]))
print(model)        