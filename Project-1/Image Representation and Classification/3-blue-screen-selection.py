import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/pizza-bluescreen.jpg')
print("This image is", type(image)," with dimensions ",image.shape)

image_copy = np.copy(image)

#Change from BGR to RGB
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

#Define color threshold
lower_blue = np.array([0,0,200])    # No red, no green, blue with intensity somewhat high
upper_blue = np.array([50,50,255])
# Define the masked area
mask = cv2.inRange(image_copy, lower_blue, upper_blue)

# Vizualize the mask
plt.imshow(mask, cmap='gray')
# Mask the image to let the pizza show through
masked_image = np.copy(image_copy)

masked_image[mask != 0] = [0, 0, 0]

# Display it!
plt.imshow(masked_image)

# Load in a background image, and convert it to RGB
background_image = cv2.imread('images/space_background.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

# Crop it to the right size (514x816)
crop_background = background_image[0:720, 0:1280]

# Mask the cropped background so that the pizza area is blocked
crop_background[mask == 0] = [0, 0, 0]

# Display the background
plt.imshow(crop_background)

# Add the two images together to create a complete image!
complete_image = masked_image + crop_background

# Display the result
plt.imshow(complete_image)
plt.show()
