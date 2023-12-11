import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU1.jpg')

# Split the image into R, G, and B channels
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))

# Create a subplot
plt.figure(figsize=(12, 4))

# Original image
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Red channel
plt.subplot(1, 4, 2)
plt.imshow(r, cmap='gray')
plt.title('Red Channel')
plt.axis('off')

# Green channel
plt.subplot(1, 4, 3)
plt.imshow(g, cmap='gray')
plt.title('Green Channel')
plt.axis('off')

# Blue channel
plt.subplot(1, 4, 4)
plt.imshow(b, cmap='gray')
plt.title('Blue Channel')
plt.axis('off')

plt.show()
