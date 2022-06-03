def abspath(path):
    import os
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)


def test_image_nuclei_2d(return_mask=False):
    """ Fluorescence microscopy image and mask from the 2018 kaggle DSB challenge

    Caicedo et al. "Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl." Nature methods 16.12
    """
    from tifffile import imread
    img = imread(abspath("images/img2d.tif"))
    mask = imread(abspath("images/mask2d.tif"))
    if return_mask:
        return img, mask
    else:
        return img


def test_image_he_2d():
    """ H&E stained RGB example image from the Cancer Imaging Archive
    https://www.cancerimagingarchive.net
    """
    from imageio import imread
    img = imread(abspath("images/histo.jpg"))
    return img


def test_image_nuclei_3d(return_mask=False):
    """ synthetic nuclei 
    """
    from tifffile import imread
    img = imread(abspath("images/img3d.tif"))
    mask = imread(abspath("images/mask3d.tif"))
    if return_mask:
        return img, mask
    else:
        return img
