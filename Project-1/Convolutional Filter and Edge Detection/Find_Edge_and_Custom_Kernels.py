import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

image = mpimg.imread('images/curved_lane.jpg')
# plt.imshow(image)
# plt.show()

#Convert to grayscale

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)
# plt.show()

#create Custome kernels
sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])
sobel_y = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])

filtered_image_x = cv2.filter2D(gray,-1,sobel_x)
filtered_image_y = cv2.filter2D(gray,-1,sobel_y)

f, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
ax1.set_title('Sobel_X')
ax1.imshow(filtered_image_x,cmap='gray')

ax2.set_title('Sobel_Y')
ax2.imshow(filtered_image_y,cmap='gray')
plt.show()
