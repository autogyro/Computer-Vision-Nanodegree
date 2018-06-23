import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/chess.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

gray = np.float32(gray)

#Detect Corner
dst = cv2.cornerHarris(gray,2,3,0.04)
# Dilate corner image to enhance corner points
dst = cv2.dilate(dst,None)

threshold = 0.0inCo1*dst.max()

corner_image = np.copy(image)
for i in range(0,dst.shape[0]):
    for j in range(0, dst.shape[1]):
        if dst[i][j] > threshold:
            cv2.circle(corner_image, (j,i), 1,(0,255,0), 3)

plt.imshow(corner_image)
plt.show()
