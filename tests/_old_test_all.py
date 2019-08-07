from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import os
import numpy as np
import pytest
from joblib import Parallel, delayed

from stardist.utils import _py_star_dist, _ocl_star_dist, _cpp_star_dist
from stardist.utils import _edt_prob, edt_prob
from stardist.utils import _fill_label_holes, fill_label_holes, dist_to_coord
from stardist.nms import non_maximum_suppression


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


def random_images(n=1,**kwargs):
    for _ in range(n):
        yield random_image(**kwargs)


def real_images():
    from tifffile import imread
    img = imread(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data','mask_ee927e8255096971ddae1bd975cf80c4ad7c847c82d0b5f5dd2ddfe5407007ee.tif'
    ))
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


@pytest.mark.parametrize('images', (real_images(),random_images(10)))
def test_edt_prob(images):
    for img in images:
        ref = _edt_prob(img)
        res = edt_prob(img)
        check_similar(ref,res)


@pytest.mark.parametrize('images', (real_images(),random_images(10)))
def test_fill_label_holes(images):
    for img in images:
        img[np.random.uniform(size=img.shape)>0.9] = 0
        ref = _fill_label_holes(img)
        res = fill_label_holes(img)
        check_similar(ref,res)


@pytest.mark.parametrize('n_rays', (4,8,16,32,64,128))
@pytest.mark.parametrize('images', (real_images(),random_images(10)))
def test_nms(images, n_rays):
    for img in images:
        prob = edt_prob(img)
        dist = _cpp_star_dist(img, n_rays)
        coord = dist_to_coord(dist)
        nms_a = non_maximum_suppression(coord, prob, prob_thresh=0.4, verbose=False, max_bbox_search=False)
        nms_b = non_maximum_suppression(coord, prob, prob_thresh=0.4, verbose=False, max_bbox_search=True)
        check_similar(nms_a,nms_b)
