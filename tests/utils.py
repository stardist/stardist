import os
import numpy as np
from tifffile import imread
from skimage.measure import label
from scipy.ndimage.filters import gaussian_filter
from pathlib import Path
from timeit import default_timer
from csbdeep.utils.tf import keras_import
Sequence = keras_import('utils','Sequence')


class NumpySequence(Sequence):
    def __init__(self, data):
        self.data = data
    def __getitem__(self,n):
        return self.data[n]
    def __len__(self):
        return len(self.data)


class Timer(object):
    def __init__(self, message = "elapsed", fmt= " {1000*t:.2f} ms"):
        self.message = message
        self.fmt = fmt
    def __enter__(self):
        self.start = default_timer()
        return self
    def __exit__(self, *args):
        t = default_timer() - self.start
        print(eval(f'f"{self.message}: {self.fmt}"'))
        self.t = t



def random_image(shape=(128, 128)):
    img = gaussian_filter(np.random.normal(size=shape), min(shape) / 20)
    img = img > np.percentile(img, 80)
    img = label(img)
    img[img > 255] = img[img > 255] % 254 + 1
    return img


def circle_image(shape=(128, 128), radius = None, center=None, eps = None):
    if center is None:
        center = (0,)*len(shape)
    if radius is None:
        radius = min(shape)//4
    if eps is None:
        eps = (1,)*len(shape)
    assert len(shape)==len(eps)
    xs = tuple(np.arange(s)-s//2 for s in shape)
    Xs = np.meshgrid(*xs, indexing="ij")
    R = np.sqrt(np.sum([(X - c) ** 2/_eps**2 for X, c,_eps in zip(Xs, center,eps)], axis=0))
    img = (R < radius).astype(np.uint16)
    return img


def overlap_image(shape=(128, 128)):
    img1 = circle_image(shape, center=(0.1,) * len(shape))
    img2 = circle_image(shape, center=(-0.1,) * len(shape))
    img = np.maximum(img1, 2 * img2)
    overlap = np.count_nonzero(np.bitwise_and(img1 > 0, img2 > 0))
    A1 = np.count_nonzero(img1 > 0)
    A2 = np.count_nonzero(img2 > 0)

    iou = overlap / min(A1, A2)
    return img, iou


def _root_dir():
    return os.path.dirname(os.path.abspath(__file__))


def real_image2d():
    img = imread(os.path.join(_root_dir(), 'data', 'img2d.tif'))
    mask = imread(os.path.join(_root_dir(), 'data', 'mask2d.tif'))
    return img, mask


def real_image3d():
    img = imread(os.path.join(_root_dir(), 'data', 'img3d.tif'))
    mask = imread(os.path.join(_root_dir(), 'data', 'mask3d.tif'))
    return img, mask


def check_similar(x, y):
    delta = np.abs(x - y)
    debug = 'avg abs err = %.10f, max abs err = %.10f' % (
        np.mean(delta), np.max(delta))
    assert np.allclose(x, y), debug


def path_model2d():
    return Path(_root_dir()) / '..' / 'models' / 'examples' / '2D_demo'


def path_model3d():
    return Path(_root_dir()) / '..' / 'models' / 'examples' / '3D_demo'


