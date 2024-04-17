from skimage.io import imread
import numpy as np

# Constants for linearization (replace with actual values from dcraw output)
black_level = 0
white_level = 16383  

# Constants for white balancing (replace with actual values from dcraw output)
r_scale = 1.628906  
g_scale = 1.000000  
b_scale = 1.386719  

image_path = '/Users/jnana/Desktop/590_ISP_ASSN3/data/baby.tiff'
image = imread(image_path)

# Report image details
print(f"Image shape (Height, Width): {image.shape}")
print(f"Bits per pixel: {image.dtype}")

# Convert image to double-precision array
image_double = image.astype(np.float64)

# Linearization
image_linear = (image_double - black_level) / (white_level - black_level)
image_linear = np.clip(image_linear, 0, 1)

# White balancing
def white_world(image):
    avg_color = np.mean(image, axis=(0, 1))
    scale = np.max(avg_color) / avg_color
    return np.clip(image * scale, 0, 1)

def gray_world(image):
    avg_color = np.mean(image, axis=(0, 1))
    scale = np.mean(avg_color) / avg_color
    return np.clip(image * scale, 0, 1)

def custom_white_balance(image, r_scale, g_scale, b_scale):
    scale = np.array([r_scale, g_scale, b_scale])
    return np.clip(image * scale, 0, 1)

# Choose one of the white balancing methods
# image_wb = white_world(image_linear)
# image_wb = gray_world(image_linear)
image_wb = custom_white_balance(image_linear, r_scale, g_scale, b_scale)

# Continue with the rest of the image processing steps...
