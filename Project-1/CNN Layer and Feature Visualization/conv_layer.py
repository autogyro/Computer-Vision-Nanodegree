import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = 'images/udacity_sdc.png'

#load color image
bgr_img = cv2.imread(img_path)
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
#normalize, rescale entries to lie in [0,1]
gray_img = gray_img.astype("float32")/255
filter_vals = np.array([[-1,-1,1,1],[-1,-1,1,1],[-1,-1,1,1],[-1,-1,1,1]])

print('Filter Shape: ', filter_vals.shape)

#defining 4 filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

print('Filters Shape:\n', filters.shape)

##Defining Convolutional layer
import torch
import torch.nn as nn
import torch.nn.functional as F



#Define a NN with a single convolutional layer and four filters
class Net(nn.Module):
    
    def __init__(self, weight):
        super(Net, self).__init__()
        #Initialize the wieghts of the convolutional layer to be the weights of 4 defined filters
        k_height, k_width = weight.shape[2:]
        #assume there are 4 grayscale filters
        self.conv = nn.Conv2d(1,4, kernel_size=(k_height, k_width), bias= False)
        self.conv.weight = torch.nn.Parameter(weight)
        
    def forward(self, x):
        #calculate output of a convolutional layer
        #pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        
        #return both layer
        return conv_x, activated_x

#Instantiate the model and set the wights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)
print(model)

# helper function for visualizing the output of a given layer
# default number of filters is 4
def viz_layer(layer, n_filters=4):
    fig = plt.figure(figsize=(20,20))
    
    for i in range(n_filters):
        ax = fig.add_subplot(1,n_filters,i+1,xticks=[], yticks=[])
        #grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s'%str(i+1))
        
#plot original image
plt.imshow(gray_img,cmap='gray')

#visualize all filters
fig = plt.figure(figsize=(12,6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8,top=1, hspace=0.05, wspace=0.05)

for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))

#convert image to input tensor
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

#get the convolytional layer (pre and post activation)
conv_layer, activated_layer = model(gray_img_tensor)

#visualize the output of a conv layer
viz_layer(conv_layer) 

# visualize the output of an activated conv layer
viz_layer(activated_layer)
    