import cv2
import numpy as np
from matplotlib import pyplot as plt

imgIn = cv2.imread('ATU1.jpg',)

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

# Harris corner detection
dst = cv2.cornerHarris(imgGrey, blockSize, aperture_size, k)

# Dilate the detected corners to make them more visible
dst = cv2.dilate(dst, None)

# Create a copy of the original image to mark the corners on
corner_img = imgIn.copy()

# Define a threshold for corner detection
threshold = 0.01 * dst.max()

# Mark the corners on the original image
corner_img[dst > threshold] = [0, 0, 255]  # Red color for corners

# Create subplot for the corner-marked image
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.xticks([]), plt.yticks([])

plt.show()
