import numpy as np
from stardist import star_dist
import pytest
from utils import random_image, real_image2d, check_similar


@pytest.mark.parametrize('img', (real_image2d()[1], random_image((128, 123))))
@pytest.mark.parametrize('n_rays', (4, 16, 32))
def test_types(img, n_rays):
    mode = "cpp"
    gt = star_dist(img, n_rays=n_rays, mode=mode)
    for dtype in (np.int8, np.int16, np.int32,
                  np.uint8, np.uint16, np.uint32):
        x = star_dist(img.astype(dtype), n_rays=n_rays, mode=mode)
        print("test_stardist2D (mode {mode}) for shape {img.shape} and type {dtype}".format(
            mode=mode, img=img, dtype=dtype))
        check_similar(gt, x)


@pytest.mark.gpu
@pytest.mark.parametrize('img', (real_image2d()[1], random_image((128, 123))))
@pytest.mark.parametrize('n_rays', (4, 16, 32))
def test_types_gpu(img, n_rays):
    mode = "opencl"
    gt = star_dist(img, n_rays=n_rays, mode=mode)
    for dtype in (np.int8, np.int16, np.int32,
                  np.uint8, np.uint16, np.uint32):
        x = star_dist(img.astype(dtype), n_rays=n_rays, mode=mode)
        print("test_stardist2D with mode {mode} for shape {img.shape} and type {dtype}".format(
            mode=mode, img=img, dtype=dtype))
        check_similar(gt, x)


@pytest.mark.gpu
@pytest.mark.parametrize('img', (real_image2d()[1], random_image((128, 123))))
@pytest.mark.parametrize('n_rays', (4, 16, 32))
def test_cpu_gpu(img, n_rays):
    s_cpp = star_dist(img, n_rays=n_rays, mode="cpp")
    s_ocl = star_dist(img, n_rays=n_rays, mode="opencl")
    check_similar(s_cpp, s_ocl)


if __name__ == '__main__':
    pass
