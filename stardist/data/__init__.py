from ..utils import abspath 

def test_image_nuclei_2d():
    from tifffile import imread
    img = imread(abspath(__file__,"images/dsb_test_image.tif"))
    return img


def test_image_nuclei_3d():
    from tifffile import imread
    img = imread(abspath(__file__,"images/test_stack_3d.tif"))
    return img
