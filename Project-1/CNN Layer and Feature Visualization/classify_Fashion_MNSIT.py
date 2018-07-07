import torch, torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

data_transform = transforms.ToTensor()
train_data = FashionMNIST(root='./data', train=True,download=False, transform=data_transform)

test_data = FashionMNIST(root='./data', train=False, download=False, transform=data_transform)

print('Train data, number of images', len(train_data))
print('Test data, number of images', len(test_data))

batch_size = 20
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#iterate over random batch and display training data
import numpy as np
import matplotlib.pyplot as plt

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25,4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(classes[labels[idx]])
    
#Defining Neural Network layers
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10,20,3)
        self.fc1 = nn.Linear(20*5*5, 50)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(50,10)
    
    def forward(self, x):
        #two conv/relu + pool layes
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        #prep for Linear layer
        #Flattening in pytorch
        x = x.view(x.size(0),-1)
        
        #two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        return x

net = Net()
print(net)

#Specifying loss function and optimizer
import torch.optim as optim

#criterion = nn.NLLLoss()
#cross entropy loss combines softmax and nn.NLLLoss() in one single class.
criterion = nn.CrossEntropyLoss()
# stochastic gradient descent with a small learning rate AND some momentum
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#calculate accuracy
correct = 0
total = 0

for images, labels in test_loader:
    outputs = net(images)
    #get the predicted class from the maximum value in the output-list of class scores
    _, predicted = torch.max(outputs.data,1)
    
    total += labels.size(0)
    correct += (predicted==labels).sum()

accuracy = 100.0*correct.item() / total

print('Accuracy before training: ',accuracy)

#Training the network
def train(n_epochs):
    
    loss_over_time = [] #track loss as network trains
    for epoch in range(n_epochs):
        running_loss = 0.0
        
        for batch_i, data in enumerate(train_loader):
            inputs, labels = data
            
            #zero th parameter gradients
            optimizer.zero_grad()
            
            #forward pass to get outputs
            outputs = net(inputs)
            
            #calculate the loss
            loss = criterion(outputs, labels)
            
            # backward pass to calculate the parameter gradients
            loss.backward()
            
            #update the parameters
            optimizer.step()
            
            running_loss += loss.item()
            
            #print avg loss in every 1000 batches
            if batch_i % 1000 ==999:
                avg_loss = running_loss/1000
                loss_over_time.append(avg_loss)
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch+1, batch_i+1,avg_loss))
                running_loss = 0.0
        
    print('Finished Traning')
    return loss_over_time

n_epochs = 10
training_loss = train(n_epochs)

# visualize the loss as the network trained
plt.plot(training_loss)
plt.xlabel('1000\'s of batches')
plt.ylabel('loss')
plt.ylim(0, 2.5) # consistent scale
plt.show()

##Test the trained network
# initialize tensor and lists to monitor test loss and accuracy
test_loss = torch.zeros(1)
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

#set the module to evaluate mode
net.eval()

for batch_i, data in enumerate(test_loader):
    
    inputs, labels = data
    outputs = net(inputs)
    
    loss = criterion(outputs, labels)
    test_loss +=(torch.ones(1)/(batch_i+1)) * (loss.data- test_loss)
    _, predicted = torch.max(outputs.data, 1)
    correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
    
    # calculate test accuracy for *each* object class
    # we get the scalar value of correct items for a class, by calling `correct[i].item()`
    for i in range(batch_size):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))

for i in range(10):
    if class_total[i]>0:
        print('Test accuracy of %5s: %2d %% (%2d/%2d)'%(
                classes[i], 100*class_correct[i]/class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training example)' %(classes[i]))
        
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)'%(
        100. * np.sum(class_correct)/np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

#Visualize sample test results
dataiter = iter(test_loader)
images, labels = dataiter.next()

preds = np.squeeze(net(images).data.max(1, keepdim=True)[1].numpy())
images = images.numpy()

fig = plt.figure(figsize=(25,4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx] else "red"))
    
#Saving the model
model_dir = 'saved_models/'
model_name = 'fashion_net_ex.pt'

import os
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

torch.save(net.state_dict(), model_dir+model_name)


##Load a trained, saved model
#instantiate our Net
#this refers to out Net class defined above
net = Net()
net.load_state_dict(torch.load('saved_models/fashion_net_ex.pt'))
print(net)


x1 = torch.zeros((10,10))
x1 = x1.unsqueeze(1).unsqueeze(0)
print(x1.shape)