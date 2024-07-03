import numpy as np

def white_world(image):
    max_value = np.max(image, axis=(0, 1))
    return image / max_value

def gray_world(image):
    mean_value = np.mean(image, axis=(0, 1))
    return image / mean_value

def custom_white_balance(image, r_scale, g_scale, b_scale):
    scale = np.array([r_scale, g_scale, b_scale])[None, None, :]
    return np.clip(image * scale, 0, 1)