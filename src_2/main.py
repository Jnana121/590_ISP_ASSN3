from python_initials import python_initials
from identify_bayer_pattern import identify_bayer_pattern
from linearization import linearize_image
from demosaicing import demosaic
from white_balancing import white_world, gray_world, custom_white_balance
from color_space_correction import apply_color_space_correction
from brightness_gamma import adjust_brightness_and_gamma
from compression import save_image

def main():
    image_path = '/Users/jnana/Desktop/590_ISP_ASSN3/data/baby.tiff'
    raw_image_path = '/Users/jnana/Desktop/590_ISP_ASSN3/data/baby.nef'
    black_level = 0
    white_level = 16383
    r_scale = 1.628906
    g_scale = 1.000000
    b_scale = 1.386719
    M_sRGB_to_cam = [[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]]

    # Read and process the image
    image_double = python_initials(image_path)

    # Identify the Bayer pattern
    bayer_pattern_str = identify_bayer_pattern(raw_image_path)
    print(bayer_pattern_str)

    # Linearize, demosaic, and white balance the image
    image_linear = linearize_image(image_double, black_level, white_level)
    image_rgb = demosaic(image_linear, bayer_pattern_str)
    image_ww = white_world(image_rgb)
    image_gw = gray_world(image_rgb)
    image_custom = custom_white_balance(image_rgb, r_scale, g_scale, b_scale)

    # Apply color space correction and brightness/gamma adjustments
    image_corrected_ww = apply_color_space_correction(image_ww, M_sRGB_to_cam)
    image_corrected_gw = apply_color_space_correction(image_gw, M_sRGB_to_cam)
    image_corrected_custom = apply_color_space_correction(image_custom, M_sRGB_to_cam)

    final_image_ww = adjust_brightness_and_gamma(image_corrected_ww)
    final_image_gw = adjust_brightness_and_gamma(image_corrected_gw)
    final_image_custom = adjust_brightness_and_gamma(image_corrected_custom)

    # Save the final images
    save_image(final_image_ww, 'final_image_white_world')
    save_image(final_image_gw, 'final_image_gray_world')
    save_image(final_image_custom, 'final_image_custom_white_balance')

if __name__ == "__main__":
    main()