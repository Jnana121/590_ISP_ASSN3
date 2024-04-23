from skimage.io import imread
from python_initials import python_initials
from identify_bayer_pattern import identify_bayer_pattern
from linearization import linearize_image
from demosaicing import demosaic
from white_balancing import custom_white_balance
from color_space_correction import apply_color_space_correction
from brightness_gamma import adjust_brightness_and_gamma
from compression import save_image

def main():
    image_path = '/Users/jnana/Desktop/590_ISP_ASSN3/data/baby.tiff'
    raw_image_path = '/Users/jnana/Desktop/590_ISP_ASSN3/data/baby.nef'
    black_level = 0
    white_level = 16383
    r_scale = 2.214791
    g_scale = 1.000000
    b_scale = 1.193155
    M_sRGB_to_cam = [2.214791, 1.000000, 1.193155, 1.000000]

    image_double = python_initials(image_path)
    bayer_pattern_array = identify_bayer_pattern(raw_image_path)
    image_linear = linearize_image(image_double, black_level, white_level)
    image_rgb = demosaic(image_linear, 'rggb')  # Update 'rggb' based on actual Bayer pattern
    image_wb = custom_white_balance(image_rgb, r_scale, g_scale, b_scale)
    image_corrected = apply_color_space_correction(image_wb, M_sRGB_to_cam)
    final_image = adjust_brightness_and_gamma(image_corrected)

    save_image(final_image, 'final_image')

if __name__ == "__main__":
    main()