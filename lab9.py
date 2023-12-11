import cv2
import numpy as np
from matplotlib import pyplot as plt

imgIn = cv2.imread('ATU1.jpg')

# Greyscale image
imgGrey = cv2.cvtColor(imgIn, cv2.COLOR_BGR2GRAY)

# Create copies of the original image
imgHarris = imgIn.copy()
imgShiTomasi = imgIn.copy() 

# Create a figure to display the images
plt.figure(figsize=(12, 8))

# Create subplot
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(imgIn, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(imgGrey, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Greyscale Image')
plt.xticks([]), plt.yticks([])

blockSize = 2
aperture_size = 3
k = 0.04

# Perform Harris corner detection
dst = cv2.cornerHarris(imgGrey, blockSize, aperture_size, k)

# Dilate the detected corners to make them more visible
dst = cv2.dilate(dst, None)

# Threshold for corner detection
threshold = 0.01

# Get the maximum value in the dst matrix
max_value = dst.max()

# Loop through the dst matrix and draw circles on the image for corners
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold * max_value):
            cv2.circle(imgHarris, (j, i), 3, (0, 0, 255), -1)

# Plot Harris corners
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.xticks([]), plt.yticks([])

# Perform Shi-Tomasi corner detection 
maxCorners = 100 
qualityLevel = 0.01
minDistance = 10

corners = cv2.goodFeaturesToTrack(imgGrey, maxCorners, qualityLevel, minDistance)

# Loop through the corners and draw circles on the image
for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(int(x),int(y)),3,(0, 0, 255),-1)

# Plot Shi-Tomasi image
plt.subplot(2, 3, 4)
plt.imshow(imgShiTomasi, cmap='gray')
plt.title('Shi-Tomasi Corner Detection')
plt.xticks([]), plt.yticks([])

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(imgIn,None)
# compute the descriptors with ORB
kp, des = orb.compute(imgIn, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(imgIn, kp, None, color=(0,255,0), flags=0)
plt.subplot(2, 3, 5)
plt.imshow(img2)
plt.title('ORB Keypoints')
plt.xticks([]), plt.yticks([])

plt.show()