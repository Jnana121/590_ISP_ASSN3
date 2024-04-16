from skimage.io import imread
import numpy as np

# Constants for linearization (replace with actual values from dcraw output)
black_level = 0  # Replace <black> with the actual black level
white_level = 16383  # Replace <white> with the actual white level

image_path = '/Users/jnana/Desktop/590_ISP_ASSN3/data/baby.tiff'
image = imread(image_path)

# Report image details
print(f"Image shape (Height, Width): {image.shape}")
print(f"Bits per pixel: {image.dtype}")

# Convert image to double-precision array
image_double = image.astype(np.float64)

# Linearization
# Map the black level to 0 and the white level to 1
image_linear = (image_double - black_level) / (white_level - black_level)
# Clip values to the range [0, 1]
image_linear = np.clip(image_linear, 0, 1)

# Continue with the rest of the image processing steps...
