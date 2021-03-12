def abspath(path):
    import os
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)


def test_image_nuclei_2d(return_mask=False):
    from tifffile import imread
    img = imread(abspath("images/img2d.tif"))
    mask = imread(abspath("images/mask2d.tif"))
    if return_mask:
        return img, mask
    else:
        return img


def test_image_nuclei_3d(return_mask=False):
    from tifffile import imread
    img = imread(abspath("images/img3d.tif"))
    mask = imread(abspath("images/mask3d.tif"))
    if return_mask:
        return img, mask
    else:
        return img
