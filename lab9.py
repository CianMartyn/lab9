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

plt.show()