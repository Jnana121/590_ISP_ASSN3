from skimage.io import imread
import numpy as np

image_path = '/Users/jnana/Desktop/590_ISP_ASSN3/data/baby.tiff'
image = imread(image_path)

# Report image details
print(f"Image shape (Height, Width): {image.shape}")
print(f"Bits per pixel: {image.dtype}")

# Convert image to double-precision array
image_double = image.astype(np.float64)