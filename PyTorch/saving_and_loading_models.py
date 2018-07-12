import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import helper
import fc_model

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Create the network, define the criterion and optimizer
model = fc_model.Network(784, 10, [512, 256, 128])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=2)

#The parameters for PyTorch networks are stored in a model's state_dict. 
#We can see the state dict contains the weight and bias matrices for each of our layers.

print("Our model : \n\n",model, '\n')
print("The state dict keys", model.state_dict().keys())

#save model with name checkpoint.pt
torch.save(model.state_dict(), 'checkpoint.pth')

#load model
state_dict = torch.load('checkpoint.pt')
print(state_dict.keys())

#But it creates a problem here, if model architecture is different. Like the example below:
'''
model = fc_model.Network(784,10,[400,200,100])
model.load_state_dict(state_dict)
'''
# So build a dictionary for checkpoint
checkpoint = {'input_size':784,
              'output_size':10,
              'hidden_layers':[each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}
torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint('checkpoint.pth')
print(model)
