from skimage.io import imsave
import numpy as np

def save_image(image, filename, quality=95):
    # Convert image from float64 to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Save as PNG (lossless)
    imsave(f'{filename}.png', image_uint8)
    
    # Save as JPEG (lossy, with quality parameter)
    imsave(f'{filename}.jpg', image_uint8, quality=quality)
