import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/green-car.jpg')
print(image.shape)

image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

lower_green = np.array([0,0,0])
upper_green = np.array([50,255,70])

mask = cv2.inRange(image_copy, lower_green, upper_green)
plt.imshow(mask)
plt.show()
image_copy[mask!=0] = [0,0,0]

background_image = cv2.imread('images/space_background.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

crop_background = background_image[0:720,0:1280]
crop_background[mask==0] = [0,0,0]
    
complete_image = image_copy + crop_background
plt.imshow(complete_image)
plt.show()
