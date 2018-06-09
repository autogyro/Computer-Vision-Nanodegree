import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/multi_faces.jpg')
image_copy = np.copy(image)

gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap = 'gray')
plt.show()

#How many faces are detected is determined by the function,
#detectMultiScale which aims to detect faces of varying sizes.
#The inputs to this function are: (image, scaleFactor, minNeighbors);
#you will often detect more faces with a smaller scaleFactor,
#and lower value for minNeighbors, but raising these values often produces better matches.
#Modify these values depending on your input image.

#load in cascade classifier
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

#Run the detector on grayscale image
faces = face_cascade.detectMultiScale(gray,4,6)
#The output of the classifier is an array of detections;coordinates that define the dimensions of a bounding box around each face.
#Note that this always outputs a bounding box that is square in dimension.

# print out the detections found
print ('We found ' + str(len(faces)) + ' faces in this image')
print ("Their coordinates and lengths/widths are as follows")
print ('=============================')
print (faces)

img_with_detections = np.copy(image)
# loop over our detections and draw their corresponding boxes on top of our original image
for (x,y,w,h) in faces:
    # draw next detection as a red rectangle on top of the original image.
    # Note: the fourth element (255,0,0) determines the color of the rectangle,
    # and the final argument (here set to 5) determines the width of the drawn rectangle
    cv2.rectangle(img_with_detections,(x,y),(x+w,y+h),(255,0,0),5)

#display result
plt.figure(figsize=(20,10))
plt.imshow(img_with_detections)
plt.show()
