import cv2
import numpy as np
from matplotlib import pyplot as plt

imgIn = cv2.imread('ATU1.jpg')

# Create a deep copy of the original image
imgHarris = imgIn.copy()

imgGrey = cv2.cvtColor(imgIn, cv2.COLOR_BGR2GRAY)

# Create a figure to display the images
plt.figure(figsize=(12, 8))

# Create subplot: 2 rows, 2 columns
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

# Define a threshold for corner detection (e.g., 0.01)
threshold = 0.01

# Get the maximum value in the dst matrix
max_value = dst.max()

# Loop through the dst matrix and draw circles on the image for corners
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold * max_value):
            # Set appropriate B, G, R values for the circle color (e.g., Red)
            cv2.circle(imgHarris, (j, i), 3, (0, 0, 255), -1)

# Create subplot for the corner-marked image
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.xticks([]), plt.yticks([])

plt.show()
