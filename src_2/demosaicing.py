import numpy as np
from scipy.interpolate import RegularGridInterpolator

def demosaic(image_linear, bayer_pattern):
    # Extract the color channels from the Bayer pattern
    # The bayer_pattern should be a string like 'rggb', 'bggr', etc.
    # Implement the extraction based on the provided bayer_pattern
    # For simplicity, here's an example assuming 'rggb'
    red_channel = image_linear[0::2, 0::2]
    green_channel_r = image_linear[0::2, 1::2]
    green_channel_b = image_linear[1::2, 0::2]
    blue_channel = image_linear[1::2, 1::2]

    # Interpolate the missing values in each color channel
    # Implement interpolation based on the provided bayer_pattern
    # For simplicity, here's an example assuming 'rggb'
    interpolate_red = RegularGridInterpolator((np.arange(red_channel.shape[0]), np.arange(red_channel.shape[1])), red_channel)
    interpolate_blue = RegularGridInterpolator((np.arange(blue_channel.shape[0]), np.arange(blue_channel.shape[1])), blue_channel)

    x_new = np.linspace(0, red_channel.shape[1]-1, image_linear.shape[1])
    y_new = np.linspace(0, red_channel.shape[0]-1, image_linear.shape[0])
    x_mesh, y_mesh = np.meshgrid(x_new, y_new, indexing='xy')

    red_interpolated = interpolate_red((y_mesh, x_mesh))
    blue_interpolated = interpolate_blue((y_mesh, x_mesh))
    green_interpolated = (green_channel_r + green_channel_b) / 2

    return np.dstack((red_interpolated, green_interpolated, blue_interpolated))