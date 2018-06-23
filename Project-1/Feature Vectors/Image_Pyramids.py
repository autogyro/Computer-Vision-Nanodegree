import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('images/cat.jpeg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

level1 = cv2.pyrDown(image)
level2 = cv2.pyrDown(level1)
level3 = cv2.pyrDown(level2)

f, (ax1,ax2,ax3,ax4) =  plt.subplots(1,4, figsize=(20,10))

ax1.set_title("Original")
ax1.imshow(image)

ax2.set_title("Level 1")
ax2.imshow(level1)
#Set the data limits for the x-axis and y-axis
ax2.set_xlim([0,image.shape[1]])
ax2.set_ylim([image.shape[0],0])

ax3.set_title("Level 2")
ax3.imshow(level2)
ax3.set_xlim([0,image.shape[1]])
ax3.set_ylim([image.shape[0],0])

ax4.set_title("Level 3")
ax4.imshow(level3)
ax4.set_xlim([0,image.shape[1]])
ax4.set_ylim([image.shape[0],0])

plt.show()
