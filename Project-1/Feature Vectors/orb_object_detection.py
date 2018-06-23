import cv2
import matplotlib.pyplot as plt
import copy

plt.rcParams['figure.figsize'] = [17.0, 17.0]

image1 = cv2.imread('images/face.jpeg')
image2 = cv2.imread('images/Team.jpeg')

training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
query_gray = cv2.cvtColor(query_image, cv2.COLOR_RGB2GRAY)

orb = cv2.ORB_create(5000, 2.0)

keypoint_train, descriptor_train = orb.detectAndCompute(training_gray, None)
keypoint_query, descriptor_query = orb.detectAndCompute(query_gray, None)

# Create copies of the query images to draw our keypoints on
query_img_keyp = copy.copy(query_image)

# Draw the keypoints with size and orientation on the copy of the query image
cv2.drawKeypoints(query_image,  keypoint_query, query_img_keyp, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.title("Keypoint with size and orientation", fontsize = 30)
plt.imshow(query_img_keyp)
plt.show()


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(descriptor_train, descriptor_query)
matches = sorted(matches, key = lambda x : x.distance)
result = cv2.drawMatches(training_gray, keypoint_train, query_gray, keypoint_query, matches[:85], query_gray, flags=2)

plt.title('Best Matching Points', fontsize = 30)
plt.imshow(result)
plt.show()

print("Number of Keypoints Detected In The Training Image: ", len(keypoint_train))

# Print the number of keypoints detected in the query image
print("Number of Keypoints Detected In The Query Image: ", len(keypoint_query))

# Print total number of matching Keypoints between the training and query images
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))