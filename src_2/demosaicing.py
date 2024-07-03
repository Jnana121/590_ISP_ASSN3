import numpy as np
from scipy.interpolate import RectBivariateSpline

def demosaic(image_linear, bayer_pattern):
    if bayer_pattern == 'rggb':
        red_channel = image_linear[0::2, 0::2]
        if red_channel.ndim > 2:
            red_channel = red_channel[:, :, 0]
        green_channel_r = image_linear[0::2, 1::2]
        if green_channel_r.ndim > 2:
            green_channel_r = green_channel_r[:, :, 0]
        green_channel_b = image_linear[1::2, 0::2]
        if green_channel_b.ndim > 2:
            green_channel_b = green_channel_b[:, :, 0]
        blue_channel = image_linear[1::2, 1::2]
        if blue_channel.ndim > 2:
            blue_channel = blue_channel[:, :, 0]
    else:
        raise ValueError("Unsupported Bayer pattern")

    height_r, width_r = red_channel.shape
    height_gr, width_gr = green_channel_r.shape
    height_gb, width_gb = green_channel_b.shape
    height_b, width_b = blue_channel.shape

    x_r = np.arange(width_r)
    y_r = np.arange(height_r)
    x_gr = np.arange(width_gr)
    y_gr = np.arange(height_gr)
    x_gb = np.arange(width_gb)
    y_gb = np.arange(height_gb)
    x_b = np.arange(width_b)
    y_b = np.arange(height_b)
    x_full = np.arange(image_linear.shape[1])
    y_full = np.arange(image_linear.shape[0])

    interpolate_red = RectBivariateSpline(y_r, x_r, red_channel)
    interpolate_green_r = RectBivariateSpline(y_gr, x_gr, green_channel_r)
    interpolate_green_b = RectBivariateSpline(y_gb, x_gb, green_channel_b)
    interpolate_blue = RectBivariateSpline(y_b, x_b, blue_channel)

    red_interpolated = interpolate_red(y_full, x_full)
    green_interpolated_r = interpolate_green_r(y_full, x_full)
    green_interpolated_b = interpolate_green_b(y_full, x_full)
    blue_interpolated = interpolate_blue(y_full, x_full)

    green_interpolated = (green_interpolated_r + green_interpolated_b) / 2

    image_rgb = np.stack((red_interpolated, green_interpolated, blue_interpolated), axis=-1)
    return image_rgb