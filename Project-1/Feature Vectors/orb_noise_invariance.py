import cv2
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [14.0, 17.0]

image1 = cv2.imread('images/face.jpeg')
image2 = cv2.imread('images/faceRN5.png')

training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
query_gray = cv2.cvtColor(query_image, cv2.COLOR_RGB2GRAY)

orb = cv2.ORB_create(1000, 2.0)

keypoint_train, descriptor_train = orb.detectAndCompute(training_gray, None)
keypoint_query, descriptor_query = orb.detectAndCompute(query_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(descriptor_train, descriptor_query)
matches = sorted(matches, key = lambda x: x.distance)

results = cv2.drawMatches(training_gray, keypoint_train, query_gray, keypoint_query, matches[:100], query_gray, flags=2)

plt.title("Best Matching Points")
plt.imshow(results)
plt.show()

print("\nNumber of Keypoints Detected In The Training Image: ", len(keypoints_train))
print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))
print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))
