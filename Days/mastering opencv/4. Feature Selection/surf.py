import cv2
import numpy as np

image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create SURF feature detector object
surf = cv2.xfeatures2d.SURF_create()

# Detect key points
keypoints = surf.detect(gray, None)
print("Number of keypoints Detected: ", len(keypoints))

# Draw rich key points on input image
image = cv2.drawKeypoints(
    image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Feature Method - SURF ", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
