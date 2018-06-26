from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import numpy as np
# import pytest
from joblib import Parallel, delayed

from stardist.utils import _py_star_dist, _ocl_star_dist, _cpp_star_dist
from stardist.utils import _edt_prob, edt_prob
from stardist.utils import _fill_label_holes, fill_label_holes


def random_image(shape=(128,128)):
    from skimage.measure import label
    from skimage.morphology import binary_closing, binary_opening
    from skimage.morphology import disk
    img = np.random.normal(size=shape)
    img = img > -0.7
    img = binary_opening(img,disk(2))
    img = binary_closing(img,disk(1))
    img = label(img)
    return img

def real_images():
    from tifffile import imread
    img = imread('data/mask_ee927e8255096971ddae1bd975cf80c4ad7c847c82d0b5f5dd2ddfe5407007ee.tif')
    yield img

def check_sd_ocl(ref,img,n_rays):
    return check_sd(_ocl_star_dist,ref,img,n_rays)

def check_sd_cpp(ref,img,n_rays):
    return check_sd(_cpp_star_dist,ref,img,n_rays)

def check_sd(dist_func,ref,img,n_rays):
    res = dist_func(img,n_rays)
    delta = np.abs(ref-res)
    debug = 'n_rays = %02d, avg abs err = %.10f, max abs err = %.10f' % (n_rays, np.mean(delta), np.max(delta))
    assert np.allclose(ref,res), debug

def test_star_dist_synth():
    tasks = [(n_rays,random_image()) for n_rays in (1,2,4,8,16,32,64,128)]
    refs = Parallel(n_jobs=-1)( delayed(_py_star_dist)(img,n_rays) for n_rays,img in tasks )
    for (n_rays,img),ref in zip(tasks,refs):
        check_sd_ocl(ref,img,n_rays)
        check_sd_cpp(ref,img,n_rays)

def test_star_dist_real():
    tasks = [(n_rays,img) for n_rays in (1,2,4,8,16,32,64,128) for img in real_images()]
    refs = Parallel(n_jobs=-1)( delayed(_py_star_dist)(img,n_rays) for n_rays,img in tasks )
    for (n_rays,img),ref in zip(tasks,refs):
        check_sd_ocl(ref,img,n_rays)
        check_sd_cpp(ref,img,n_rays)

def check_similar(ref,res):
    delta = np.abs(ref-res)
    debug = 'avg abs err = %.10f, max abs err = %.10f' % (np.mean(delta), np.max(delta))
    assert np.allclose(ref,res), debug

def test_edt_prob_synth():
    for i in range(10):
        img = random_image()
        ref = _edt_prob(img)
        res = edt_prob(img)
        check_similar(ref,res)

def test_edt_prob_real():
    for img in real_images():
        ref = _edt_prob(img)
        res = edt_prob(img)
        check_similar(ref,res)

def test_fill_label_holes_synth():
    for i in range(10):
        img = random_image()
        img[np.random.uniform(size=img.shape)>0.9] = 0
        ref = _fill_label_holes(img)
        res = fill_label_holes(img)
        check_similar(ref,res)

def test_fill_label_holes_real():
    for img in real_images():
        img[np.random.uniform(size=img.shape)>0.9] = 0
        ref = _fill_label_holes(img)
        res = fill_label_holes(img)
        check_similar(ref,res)
