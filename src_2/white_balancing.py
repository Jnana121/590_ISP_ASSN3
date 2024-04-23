import numpy as np

def custom_white_balance(image, r_scale, g_scale, b_scale):
    scale = np.array([r_scale, g_scale, b_scale])[None, None, :]
    return np.clip(image * scale, 0, 1)