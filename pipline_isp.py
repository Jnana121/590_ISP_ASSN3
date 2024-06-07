from skimage.io import imread, imsave
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import os

BLACK = 0
WHITE = 16383
RED = 1.628906
GREEN = 1.00000
BLUE = 1.386719

raw_image = imread("data/baby.tiff")
# raw_iamge.dtype.itemsize returns the number of bytes of each element in the array
# in our case it is 2 bytes representing an unsigned 16 bit integer
bits_per_pixel = raw_image.dtype.itemsize * 8
print(f"Bits Per Pixel: {bits_per_pixel}, Width: {len(raw_image[0])}, Height: {len(raw_image)}")

# no named double type but its just a floating point with 64 bits, so we sue float64
raw_image_double = raw_image.astype(np.float64)
print(f"New 2D array type: {raw_image_double.dtype}")

plt.imshow(np.clip(raw_image_double * 5, 0, WHITE), cmap="gray")
plt.colorbar()
plt.show()

def display_plot(image, name, clip, color = False):
    if color:
        plt.imshow(np.clip(image * 5, 0, clip))
    else:
        plt.imshow(np.clip(image * 5, 0, clip), cmap="gray")
    plt.title(name)
    plt.colorbar()
    plt.show()


scaled_image = (raw_image_double - BLACK) / (WHITE - BLACK)
linearized_image = np.clip(scaled_image, 0, 1)
print(f"RAW image min: {np.min(raw_image_double)}, RAW image max: {np.max(raw_image_double)}")
print(f"Linearized min: {np.min(linearized_image)}, Linearized max: {np.max(linearized_image)}")

plt.imshow(np.clip(linearized_image * 5, 0, 1), cmap="gray")
plt.colorbar()
plt.show()

four_by_four_block = linearized_image[2000:2004, 1200:1204]
print(four_by_four_block)

sampled_pixels = linearized_image[2000:2200]
even_rows = sampled_pixels[::2, 1200:2000]
odd_rows = sampled_pixels[1::2, 1200:2000]

even_rows_removed_green =  even_rows[:, ::2]
odd_rows_removed_green =  odd_rows[:, 1::2]
print("Even rows without green average: ", even_rows_removed_green.mean())
print("Odd rows without green average: ", odd_rows_removed_green.mean())

# White World White Balancing

# Find the maximum value of each color channel
red_max = np.max(linearized_image[::2, ::2])
green_max = np.max(np.maximum(linearized_image[::2, 1::2], linearized_image[1::2, ::2]))
blue_max = np.max(linearized_image[1::2, 1::2])

# Calculate the scaling factors for each color channel
max_value = np.max([red_max, green_max, blue_max])
red_scale = max_value / red_max
green_scale = max_value / green_max
blue_scale = max_value / blue_max

# Apply the scaling factors to the image
white_world_balanced_image = np.copy(linearized_image)
white_world_balanced_image[::2, ::2] *= red_scale
white_world_balanced_image[::2, 1::2] *= green_scale
white_world_balanced_image[1::2, ::2] *= green_scale
white_world_balanced_image[1::2, 1::2] *= blue_scale

print(f"Red scaling factor: {red_scale}, Green scaling factor: {green_scale}, Blue scaling factor: {blue_scale}")

# Normalize the white-balanced image back to the range [0, 1]
white_world_balanced_image = np.clip(white_world_balanced_image, 0, 1)

red_avg = np.mean(linearized_image[::2, ::2])
green_avg = np.mean(linearized_image[::2, 1::2] + linearized_image[1::2, ::2]) / 2
blue_avg = np.mean(linearized_image[1::2, 1::2])

total_avg = (red_avg + green_avg + blue_avg) / 3

grey_world_balanced_image = np.copy(linearized_image)
grey_world_balanced_image[::2, ::2] *= total_avg / red_avg
white_world_balanced_image[::2, 1::2] *= total_avg / green_avg
white_world_balanced_image[1::2, ::2] *= total_avg / green_avg
grey_world_balanced_image[1::2, 1::2] *= total_avg / blue_avg

print(f"Red scaling factor: {total_avg / red_avg}, Green scaling factor: {total_avg / green_avg}, Blue scaling factor: {total_avg / blue_avg}")

grey_world_balanced_image = np.clip(grey_world_balanced_image, 0, 1)

# white_balancing_kernel = np.array([[RED, GREEN], [GREEN, BLUE]], dtype=np.float64)
# white_balanced_image = linearized_image @ white_balancing_kernel

white_balanced_image = linearized_image.copy()
white_balanced_image[::2, ::2] = linearized_image[::2, ::2] * RED
white_balanced_image[1::2, 1::2] = linearized_image[1::2, 1::2] * BLUE

white_balanced_image

display_plot(raw_image_double, "raw image", WHITE)
display_plot(white_world_balanced_image, "white world", 1)
display_plot(grey_world_balanced_image, "gray world", 1)
display_plot(white_balanced_image, "white reconnaissance values", 1)
# fig = plt.figure()
# fig.add_subplot(2, 2, 1)
# plt.imshow(raw_image_double, cmap="gray")
# fig.add_subplot(2, 2, 2)
# plt.imshow(linearized_image, cmap="gray")
# fig.add_subplot(2, 2, 3)
# plt.imshow(white_world_balanced_image, cmap="gray")
# fig.add_subplot(2, 2, 4)
# plt.imshow(white_balanced_image, cmap="gray")
# plt.show()

bayer_pattern = "RGGB"

demosaiced_image = demosaicing_CFA_Bayer_bilinear(white_balanced_image, bayer_pattern)
demosaiced_image_2 = demosaicing_CFA_Bayer_bilinear(white_world_balanced_image, bayer_pattern)

print(demosaiced_image.shape)
print(demosaiced_image_2.shape)

display_plot(demosaiced_image, "Demosaiced Image", 1, True)
display_plot(demosaiced_image_2, "Demosaiced Image", 1, True)

sRGBtoXYZ = np.asmatrix([
    [0.4124564, 0.3575761, 0.1804375], 
    [0.2126729, 0.7151522, 0.0721750], 
    [0.0193339, 0.1191920, 0.9503041]
    ])

XYZtoCAM = np.asmatrix([
    [6988, -1384, -714], 
    [-5631, 13410, 2447], 
    [-1485, 2204, 7318]
    ])

sRGBtoCAM = XYZtoCAM @ sRGBtoXYZ

row_sums = sRGBtoCAM.sum(axis=1)
sRGBtoCAM_normalized = sRGBtoCAM / row_sums

print(sRGBtoCAM_normalized)

sRGBtoCAM_inverse= np.linalg.inv(sRGBtoCAM_normalized)

corrected_image = np.zeros_like(demosaiced_image, dtype=np.float32)

for y in range(demosaiced_image.shape[0]):
        for x in range(demosaiced_image.shape[1]):
            pixel = demosaiced_image[y, x]
            corrected_pixel = np.dot(sRGBtoCAM_inverse, pixel)
            corrected_image[y, x] = corrected_pixel

display_plot(corrected_image, "Color Correction", 1, True)
np.max(corrected_image)

def adjust_brightness(image, desired_mean=0.25):
    # Convert to grayscale and compute current mean
    grayscale = rgb2gray(image)
    current_mean = np.mean(grayscale)
    
    # Calculate scaling factor
    scaling_factor = desired_mean / current_mean
    
    # Scale and clip the image
    adjusted_image = image * scaling_factor
    adjusted_image = np.clip(adjusted_image, 0, 1)
    
    return adjusted_image

def srgb_tonemap(values):
    values = np.clip(values, 0, 1)
    mask = values <= 0.0031308
    values[mask] *= 12.92
    values[~mask] = (1 + 0.055) * np.power(values[~mask], 1/2.4) - 0.055
    return values

def gamma_encoding(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    r_encoded = srgb_tonemap(r)
    g_encoded = srgb_tonemap(g)
    b_encoded = srgb_tonemap(b)
    
    encoded_image = np.dstack((r_encoded, g_encoded, b_encoded))
    return encoded_image

adjusted_image = adjust_brightness(corrected_image)
final_image = gamma_encoding(adjusted_image)

plt.imshow(final_image)
plt.show()

print(final_image)

final_image_uint8 = (final_image * 255).astype(np.uint8)
imsave('image_recon.png', final_image_uint8)
imsave('image_recon.jpeg', final_image_uint8, format="jpeg", quality=95)

imsave("image2_recon.jpeg", final_image_uint8, format="jpeg", quality=75)

imsave("image3_recon.jpeg", final_image_uint8, format="jpeg", quality=40)

# Get the file sizes in bytes
png_size = os.path.getsize('image.png')
jpeg_size = os.path.getsize('image3.jpeg')

# Print the file sizes
print(f"PNG file size: {png_size} bytes")
print(f"JPEG file size (quality 95): {jpeg_size} bytes")

# Calculate the compression ratio
compression_ratio = png_size / jpeg_size

print(f"Compression ratio (PNG / JPEG quality 95): {compression_ratio:.2f}")

