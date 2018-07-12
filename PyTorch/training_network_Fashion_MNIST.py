from collections import OrderedDict
import helper
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

input_size =784
hidden_size = [400,200,100]
output_size = 10

model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_size[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_size[0], hidden_size[1])),
            ('relu2',nn.ReLU()),
            ('fc3', nn.Linear(hidden_size[1],hidden_size[2])),
            ('relu3', nn.ReLU()),
            ('logits',nn.Linear(hidden_size[2], output_size))]))

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02)

epoch = 4
print_every = 40
steps = 0
for e in range(epoch):
    running_loss = 0
    for images,labels in iter(trainloader):
        steps += 1
        images.resize_(images.size()[0], 784)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print("Epoch: {}/{}...".format(e+1, epoch),
                  "Loss : {:4f}".format(running_loss/print_every))
            running_loss = 0
            
dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# TODO: Calculate the class probabilities (softmax) for img
with torch.no_grad():
    logits = model.forward(img)
ps = F.softmax(logits, dim=1)

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')