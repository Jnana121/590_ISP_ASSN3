from skimage.io import imsave

def save_image(image, filename, quality=95):
    imsave(f'{filename}.png', image)  # Save as PNG (lossless)
    imsave(f'{filename}.jpg', image, quality=quality)  # Save as JPEG (lossy)