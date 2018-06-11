import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/thumbs_up_down.jpg')
image_copy = np.copy(image)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

# Create a binary thresholded image
retval, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

# plt.imshow(binary, cmap = 'gray')
# plt.show()

# Find contours from thresholded, binary image
retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_image = np.copy(image)
contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 3) #here try 1 and 0 as arguments
# plt.imshow(contours_image)
# plt.show()

def orientation(contours):
    angles = []
    for i in contours:
        (x,y), (MA,ma), angle = cv2.fitEllipse(i)
        angles.append(angle)
    return angles

angles = orientation(contours)
print('Angles of each contour (in degrees): ' + str(angles))

#Crop the image around a contour
def left_hand_crop(image, selected_contour):
    #Detect the bounding rectangle of the left hand
    x,y,w,h = cv2.boundingRect(selected_contour)
    box_image = cv2.rectangle(contours_image,(x,y),(x+w,y+h),(200,200),2)

    #Crop the image using the dimensions of the bounding rectangle
    #Make a copy of the image to crop
    cropped_image = np.copy(image)
    cropped_image = image[y:y+h,x:x+w]

    return cropped_image

selected_contour = contours[0]
if angles[1]>angles[0]:
    selected_contour = contours[1]

if selected_contour is not None:
    cropped_image = left_hand_crop(image, selected_contour)
    plt.imshow(cropped_image)
    plt.show()
