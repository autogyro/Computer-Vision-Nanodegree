import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/brain_MR.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

wide = cv2.Canny(gray,30,100)
tight = cv2.Canny(gray,200,240)

f ,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,10))

ax1.set_title('Wide Canny')
ax1.imshow(wide,cmap='gray')

ax2.set_title('Tight Canny')
ax2.imshow(tight, cmap='gray')

ax3.set_title('Original Image')
ax3.imshow(image, cmap='gray')

plt.show()
