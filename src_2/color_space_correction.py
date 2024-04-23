import numpy as np

def apply_color_space_correction(image, M_sRGB_to_cam):
    return np.dot(image, np.linalg.inv(M_sRGB_to_cam))