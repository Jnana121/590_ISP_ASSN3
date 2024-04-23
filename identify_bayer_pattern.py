import rawpy

def identify_bayer_pattern(raw_image_path):
    with rawpy.imread(raw_image_path) as raw:
        bayer_pattern_array = raw.raw_pattern
        return bayer_pattern_array
