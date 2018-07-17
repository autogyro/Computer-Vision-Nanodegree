import torch
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

data_transform = transforms.ToTensor()
test_data = FashionMNIST(root='./data', train=False, download=True, transform=data_transform)
print('Test data, number of images', len(test_data))

batch_size = 20
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

##Define network
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.pool = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(10,20,3)
        
        self.fc1 = nn.Linear(20*5*5,50)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(50,10)
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0),-1)
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        return x
    
net = Net()
net.load_state_dict(torch.load('saved_models/fashion_net_ex_withDrop.pt'))

print(net)



#### Feature Visualization ###
weights = net.conv1.weight.data
w = weights.numpy()

#fig for 10 filters
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,8))
columns = 5
rows = 2
for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(w[i][0], cmap='gray')
    
print('First Convolutional Layer')
plt.show()
