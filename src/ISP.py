from skimage.io import imread, imshow, imsave
import numpy as np
from skimage.color import rgb2gray
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# Load the image
image_path = '/Users/jnana/Desktop/590_ISP_ASSN3/data/baby.tiff'
image = imread(image_path)

# Constants for linearization
black_level = 0  # Replace with actual black level from dcraw output
white_level = 16383  # Replace with actual white level from dcraw output

# Linearization
image_linear = (image.astype(np.float64) - black_level) / (white_level - black_level)
image_linear = np.clip(image_linear, 0, 1)

# Demosaicing (assuming RGGB Bayer pattern)
red_channel = image_linear[0::2, 0::2]
green_channel_r = image_linear[0::2, 1::2]
green_channel_b = image_linear[1::2, 0::2]
blue_channel = image_linear[1::2, 1::2]

# Prepare interpolation grid
x = np.arange(0, red_channel.shape[1])
y = np.arange(0, red_channel.shape[0])
x_new = np.linspace(0, red_channel.shape[1]-1, image_linear.shape[1])
y_new = np.linspace(0, red_channel.shape[0]-1, image_linear.shape[0])
x_mesh, y_mesh = np.meshgrid(x_new, y_new, indexing='xy')

# Interpolation of the missing values in each color channel using RegularGridInterpolator
red_interpolator = RegularGridInterpolator((y, x), red_channel)
red_interpolated = red_interpolator((y_mesh, x_mesh))

blue_interpolator = RegularGridInterpolator((y, x), blue_channel)
blue_interpolated = blue_interpolator((y_mesh, x_mesh))

green_interpolated = (green_channel_r + green_channel_b) / 2
green_interpolator = RegularGridInterpolator((y, x), green_interpolated)
green_interpolated = green_interpolator((y_mesh, x_mesh))

# Stack the color channels to form the RGB image
image_rgb = np.dstack((red_interpolated, green_interpolated, blue_interpolated))

# Custom white balance
def custom_white_balance(image, r_scale, g_scale, b_scale):
    scale = np.array([r_scale, g_scale, b_scale])[None, None, :]
    return np.clip(image * scale, 0, 1)

# Apply custom white balance
image_wb = custom_white_balance(image_rgb, 1.628906, 1.000000, 1.386719)  # Update scales with actual values from dcraw output

# Convert image from float64 to uint8
image_uint8 = (image_wb * 255).astype(np.uint8)

# Display the final image
imshow(image_uint8)
plt.show()

# Save the image
imsave('final_image.png', image_uint8)
