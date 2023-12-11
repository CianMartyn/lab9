import cv2
import numpy as np
from matplotlib import pyplot as plt

imgIn1 = cv2.imread('ATU1.jpg')
imgIn2 = cv2.imread('ATU2.jpg')

# Greyscale image
imgGrey = cv2.cvtColor(imgIn1, cv2.COLOR_BGR2GRAY)

# Create copies of the original image
imgHarris = cv2.cvtColor(imgGrey, cv2.COLOR_GRAY2BGR)
imgShiTomasi = cv2.cvtColor(imgGrey, cv2.COLOR_GRAY2BGR) 

# Create a figure to display the images
plt.figure(figsize=(16, 8))

# Create subplot
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(imgIn1, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

# Plot Greyscale image
plt.subplot(2, 4, 2)
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
plt.subplot(2, 4, 3)
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
plt.subplot(2, 4, 4)
plt.imshow(imgShiTomasi, cmap='gray')
plt.title('Shi-Tomasi Corner Detection')
plt.xticks([]), plt.yticks([])

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(imgIn1,None)
# compute the descriptors with ORB
kp, des = orb.compute(imgIn1, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(imgGrey, kp, None, color=(0,255,0), flags=0)
# Plot ORB Keypoints Image
plt.subplot(2, 4, 5)
plt.imshow(img2)
plt.title('ORB Keypoints')
plt.xticks([]), plt.yticks([])

# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgIn1,None)
kp2, des2 = sift.detectAndCompute(imgIn2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(imgIn1,kp1,imgIn2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Plot BruteForceMatcher Image
plt.subplot(2, 4, 6)
plt.imshow(img2)
plt.title('Brute Force Matcher')
plt.xticks([]), plt.yticks([])
plt.imshow(img3)


# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatchesKnn(imgIn1,kp1,imgIn2,kp2,matches,None,**draw_params)
plt.imshow(img3,)

# Plot FLANN Image
plt.subplot(2, 4, 7)
plt.imshow(img2)
plt.title('Flann based Matcher')
plt.xticks([]), plt.yticks([])
plt.imshow(img3)

plt.show()