import numpy as np

def linearize_image(image, black_level, white_level):
    image_linear = (image.astype(np.float64) - black_level) / (white_level - black_level)
    return np.clip(image_linear, 0, 1)