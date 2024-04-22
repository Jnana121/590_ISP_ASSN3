import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imsave
from matplotlib.widgets import Cursor

# Load the image
image_path = '/Users/jnana/Desktop/590_ISP_ASSN3/data/baby.tiff' 
image_rgb = imread(image_path)

# Display the image and use a cursor to select the white patch
fig, ax = plt.subplots()
imshow(image_rgb)
cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
plt.show()

# Assume the coordinates of the white patch are known (e.g., selected using ginput or manually defined)
x, y = 100, 100  # Update these coordinates based on actual selection

# Extract the RGB values of the selected patch
patch_rgb = image_rgb[y-5:y+5, x-5:x+5]  # Averaging a small area around the selected point
mean_rgb = np.mean(patch_rgb, axis=(0, 1))

# Normalize the RGB channels
normalized_rgb = image_rgb / mean_rgb
normalized_rgb = (normalized_rgb / normalized_rgb.max()) * 255
normalized_rgb = normalized_rgb.astype(np.uint8)

# Save and show the result
imsave('manual_white_balanced.png', normalized_rgb)
imshow(normalized_rgb)
plt.show()
