from skimage.io import imread
import numpy as np

def python_initials(image_path):
    image = imread(image_path)
    print(f"Image shape (Height, Width): {image.shape}")
    print(f"Bits per pixel: {image.dtype}")

    # Convert image to double-precision array
    image_double = image.astype(np.float64)
    return image_double