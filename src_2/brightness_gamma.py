import numpy as np
from skimage.color import rgb2gray

def adjust_brightness_and_gamma(image, target_mean=0.25, gamma=2.4):
    grayscale = rgb2gray(image)
    current_mean = np.mean(grayscale)
    adjusted_image = image * (target_mean / current_mean)
    adjusted_image = np.clip(adjusted_image, 0, 1)

    return np.where(adjusted_image <= 0.0031308,
                    12.92 * adjusted_image,
                    1.055 * np.power(adjusted_image, 1 / gamma) - 0.055)