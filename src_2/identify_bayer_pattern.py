import rawpy

def identify_bayer_pattern(raw_image_path):
    with rawpy.imread(raw_image_path) as raw:
        bayer_pattern_array = raw.raw_pattern
        # Convert the Bayer pattern array to a string representation
        bayer_pattern_str = convert_bayer_pattern_to_string(bayer_pattern_array)
        return bayer_pattern_str

def convert_bayer_pattern_to_string(bayer_pattern_array):
    # Mapping from rawpy's numerical pattern to string pattern
    pattern_mapping = {
        0: 'r',  # Red
        1: 'g',  # Green on Red/Blue row
        2: 'b',  # Blue
        3: 'g'   # Green on Blue/Red row
    }
    pattern_str = ''.join([pattern_mapping[x] for x in bayer_pattern_array.flatten()])
    return pattern_str