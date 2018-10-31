import cv2
import numpy as np

image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create ORB , we can specify the number of key points we desire
orb = cv2.ORB_create(1000, 1.2)

# Obtain descriptors and new final keypoints using ORB
keypoints, descriptors = orb.detectAndCompute(gray, None)
print("Number of keypoints Detected: ", len(keypoints))

# Draw rich key points on input image
image = cv2.drawKeypoints(
    image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Feature Method - ORB ", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
