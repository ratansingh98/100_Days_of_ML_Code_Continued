import cv2
import numpy as np

image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create FAST feature detector object
fast = cv2.FastFeatureDetector_create()

# Create BRIEF extractor object
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Detect key points
keypoints = fast.detect(gray, None)


# Obtain descriptors and new final keypoints using BRIEF
keypoints, descriptors = brief.compute(gray, keypoints)
print("Number of keypoints Detected: ", len(keypoints))

# Draw rich key points on input image
image = cv2.drawKeypoints(
    image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Feature Method - Brief ", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
