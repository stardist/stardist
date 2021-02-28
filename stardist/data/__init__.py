

def abspath(path):
    import os
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)

def test_image_nuclei_2d():
    from tifffile import imread
    img = imread(abspath("images/dsb_test_image.tif"))
    return img
