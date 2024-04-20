from skimage.io import imread
import numpy as np
from skimage.color import rgb2gray
from scipy.interpolate import interp2d

# Constants for linearization (replace with actual values from dcraw output)
black_level = 0  # Replace <black> with the actual black level
white_level = 16383  # Replace <white> with the actual white level

# Constants for white balancing (replace with actual values from dcraw output)
r_scale = 1.628906  # Replace with actual r_scale
g_scale = 1.000000  # Replace with actual g_scale
b_scale = 1.386719  # Replace with actual b_scale

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

# Demosaicing (assuming RGGB Bayer pattern)
# Extract the color channels from the Bayer pattern
red_channel = image_linear[0::2, 0::2]  # Red channel (top-left)
green_channel_r = image_linear[0::2, 1::2]  # Green channel (top-right)
green_channel_b = image_linear[1::2, 0::2]  # Green channel (bottom-left)
blue_channel = image_linear[1::2, 1::2]  # Blue channel (bottom-right)

# Interpolate the missing values in each color channel
interpolate = interp2d(np.arange(blue_channel.shape[1]), np.arange(blue_channel.shape[0]), blue_channel, kind='linear')
blue_interpolated = interpolate(np.arange(0.5, image_linear.shape[1], 2), np.arange(0.5, image_linear.shape[0], 2))

interpolate = interp2d(np.arange(red_channel.shape[1]), np.arange(red_channel.shape[0]), red_channel, kind='linear')
red_interpolated = interpolate(np.arange(0.5, image_linear.shape[1], 2), np.arange(0.5, image_linear.shape[0], 2))

green_interpolated = (green_channel_r + green_channel_b) / 2

# Stack the color channels to form the RGB image
image_rgb = np.dstack((red_interpolated, green_interpolated, blue_interpolated))

# Custom white balance
def custom_white_balance(image, r_scale, g_scale, b_scale):
    scale = np.array([r_scale, g_scale, b_scale])[None, None, :]
    return np.clip(image * scale, 0, 1)

# Apply custom white balance
image_wb = custom_white_balance(image_rgb, r_scale, g_scale, b_scale)

# Continue with the rest of the image processing steps...
