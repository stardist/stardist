import numpy as np
from stardist import star_dist, relabel_image_stardist
import pytest
from utils import random_image, real_image2d, check_similar, circle_image


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
    print(img.shape, s_cpp.shape)
    check_similar(s_cpp, s_ocl)


@pytest.mark.parametrize('n_rays', (32,64))
@pytest.mark.parametrize('eps', ((1,1),(.4,1.3)))
def test_relabel_consistency(n_rays, eps, plot = False):
    """ test whether an already star-convex label image gets perfectly relabeld"""

    lbl1 = circle_image(shape=(32,32), radius=8, eps = eps)

    lbl2 = relabel_image_stardist(lbl1, n_rays)

    rel_error = 1-np.count_nonzero(np.bitwise_and(lbl1>0, lbl2>0))/np.count_nonzero(lbl1>0)
    print(rel_error)
    assert rel_error<1e-1

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(num=1, figsize=(8,4))
        plt.subplot(1,3,1);plt.imshow(lbl1);plt.title("GT")
        plt.subplot(1,3,2);plt.imshow(lbl2);plt.title("Reco")
        plt.subplot(1,3,3);plt.imshow(1*(lbl1>0)+2*(lbl2>0));plt.title("Overlay")
        plt.tight_layout()
        plt.show()

    return lbl1, lbl2


if __name__ == '__main__':
    lbl1, lbl2 = test_relabel_consistency(32,eps = (.7,1), plot = True)
