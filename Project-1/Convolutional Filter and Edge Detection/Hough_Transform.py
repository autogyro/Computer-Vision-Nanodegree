import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/phone.jpg')
image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

edges = cv2.Canny(gray, 50, 100)
# plt.imshow(edges, cmap = 'gray')
# plt.show()

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on

rho = 1
theta = np.pi/180
threshold = 60
min_line_length = 105
max_line_gap = 5

#creating an image copy to draw lines on
line_image = np.copy(image)
# Run Hough on the edge-detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

plt.imshow(line_image)
plt.show()
