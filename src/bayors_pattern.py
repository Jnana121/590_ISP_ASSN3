import rawpy
import numpy as np

def identify_bayer_pattern(raw_image_path):
    # Read the RAW image file
    with rawpy.imread(raw_image_path) as raw:
        # Access the raw pattern of the Bayer filter
        bayer_pattern_array = raw.raw_pattern
        # The raw_pattern attribute returns a 2x2 NumPy array indicating the CFA layout
        # You can then map these numbers to color channels based on rawpy's documentation or experimentation
        # For simplicity, this example directly returns the array
        return bayer_pattern_array

# Replace with the path to your RAW image file
raw_image_path = '/Users/jnana/Desktop/590_ISP_ASSN3/data/baby.nef'

# Identify the Bayer pattern
bayer_pattern_array = identify_bayer_pattern(raw_image_path)
print("Bayer pattern array:")
print(bayer_pattern_array)
