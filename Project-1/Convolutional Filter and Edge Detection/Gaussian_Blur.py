import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

image = mpimg.imread('images/brain_MR.jpg')
image_copy = np.copy(image)
# plt.imshow(image)
# plt.show()

#Convert to RBB
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# plt.imshow(image_copy)
# plt.show()

#Convert to grayscale for filtering
gray = cv2.cvtColor(image_copy,cv2.COLOR_RGB2GRAY)

#Create a gaussian blurred image
gray_blur = cv2.GaussianBlur(gray, (9,9),0)

# f, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
#
# ax1.set_title('Original Image')
# ax1.imshow(gray,cmap ='gray')
#
# ax2.set_title('Blurred Image')
# ax2.imshow(gray_blur,cmap='gray')
# plt.show()

#create Custome kernels
sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])
sobel_y = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])

#Filter the original and blurred grayscale images using filter2D

filtered = cv2.filter2D(gray,-1,sobel_x)
filtered_blurred = cv2.filter2D(gray_blur,-1,sobel_x)
# f, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
#
# ax1.set_title('Original Gray')
# ax1.imshow(filtered,cmap ='gray')
#
# ax2.set_title('Blurred Image')
# ax2.imshow(filtered_blurred,cmap='gray')
# plt.show()

#Create threshold that sets all the filtered pixels to white above a certain threshold

retval, binary_image = cv2.threshold(filtered_blurred,50,255,cv2.THRESH_BINARY)
plt.imshow(binary_image,cmap = 'gray')
plt.show()
