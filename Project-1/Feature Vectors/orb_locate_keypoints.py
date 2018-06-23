import matplotlib.pyplot as plt
import cv2

#Set the default figure size
plt.rcParams['figure.figsize'] = [20,10]

#Load the training image
image = cv2.imread('images/face.jpeg')
training_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
training_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Display the images
# plt.subplot(121)
# plt.title('Original Training Image')
# plt.imshow(training_image)
# plt.subplot(122)
# plt.title('Gray Scale Training Image')
# plt.imshow(training_gray, cmap = 'gray')
# plt.show()

########################Locating Keypoints###########
# Import copy to make copies of the training image
import copy
#Set the default figure size
plt.rcParams['figure.figsize'] = [14.0,7.0]

# Set the parameters of the ORB algorithm by specifying the maximum number of keypoints to locate and
# the pyramid decimation ratio
orb = cv2.ORB_create(200, 2.0)

# Find the keypoints in the gray scale training image and compute their ORB descriptor.
# The None parameter is needed to indicate that we are not using a mask.
keypoints, descriptor = orb.detectAndCompute(training_gray, None)

#Create copies of training image
keyp_without_size = copy.copy(training_image)
keyp_with_size = copy.copy(training_image)

#Draw key points without size and orientation on training_image
cv2.drawKeypoints(training_image, keypoints, keyp_without_size, color = (0,255,0))

# Draw the keypoints with size and orientation on the other copy of the training image
cv2.drawKeypoints(training_image, keypoints, keyp_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Display
plt.subplot(121)
plt.title('Keypoints without size and orientation')
plt.imshow(keyp_without_size)

plt.subplot(122)
plt.title('Keypoints with size and orientation')
plt.imshow(keyp_with_size)

plt.show()
print("\nNumber of keypoints Detected: ",len(keypoints))
