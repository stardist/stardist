import numpy as np
import pytest
from stardist import star_dist3D, Rays_GoldenSpiral, relabel_image_stardist3D
from utils import random_image, real_image3d, check_similar, circle_image


@pytest.mark.parametrize('img', (real_image3d()[1], random_image((33, 44, 55))))
@pytest.mark.parametrize('n_rays', (4, 16, 32))
@pytest.mark.parametrize('grid', ((1, 1, 1), (1, 2, 4)))
def test_types(img, n_rays, grid):
    mode = "cpp"
    rays = Rays_GoldenSpiral(n_rays)
    gt = star_dist3D(img, rays=rays, grid=grid, mode=mode)
    for dtype in (np.int8, np.int16, np.int32,
                  np.uint8, np.uint16, np.uint32):
        x = star_dist3D(img.astype(dtype), rays=rays, grid=grid, mode=mode)
        print("test_stardist3D (mode {mode}) for shape {img.shape} and type {dtype}".format(
            mode=mode, img=img, dtype=dtype))
        check_similar(gt, x)


@pytest.mark.gpu
@pytest.mark.parametrize('img', (real_image3d()[1], random_image((33, 44, 55))))
@pytest.mark.parametrize('n_rays', (4, 16, 32))
@pytest.mark.parametrize('grid', ((1, 1, 1), (1, 2, 4)))
def test_types_gpu(img, n_rays, grid):
    mode = "opencl"
    rays = Rays_GoldenSpiral(n_rays)
    gt = star_dist3D(img, rays=rays, grid=grid, mode=mode)
    for dtype in (np.int8, np.int16, np.int32,
                  np.uint8, np.uint16, np.uint32):
        x = star_dist3D(img.astype(dtype), rays=rays, grid=grid, mode=mode)
        print("test_stardist3D (mode {mode}) for shape {img.shape} and type {dtype}".format(
            mode=mode, img=img, dtype=dtype))
        check_similar(gt, x)


@pytest.mark.gpu
@pytest.mark.parametrize('img', (real_image3d()[1], random_image((33, 44, 55))))
@pytest.mark.parametrize('n_rays', (4, 16, 32))
@pytest.mark.parametrize('grid', ((1, 1, 1), (1, 2, 4)))
def test_cpu_gpu(img, n_rays, grid):
    rays = Rays_GoldenSpiral(n_rays)
    s_cpp = star_dist3D(img, rays=rays, grid=grid, mode="cpp")
    s_ocl = star_dist3D(img, rays=rays, grid=grid, mode="opencl")
    check_similar(s_cpp, s_ocl)


@pytest.mark.parametrize('n_rays', (64,128))
@pytest.mark.parametrize('eps', ((1,1,1),(.4,1.3,.7)))
def test_relabel_consistency(n_rays, eps, plot = False):
    """ test whether an already star-convex label image gets perfectly relabeld"""

    rays = Rays_GoldenSpiral(n_rays, anisotropy = 1./np.array(eps))
    # img = random_image((128, 123))
    lbl1 = circle_image(shape=(32,32,32), radius=8, eps = eps)
    
    lbl1 = relabel_image_stardist3D(lbl1, rays)

    lbl2 = relabel_image_stardist3D(lbl1, rays)

    rel_error = 1-np.count_nonzero(np.bitwise_and(lbl1>0, lbl2>0))/np.count_nonzero(lbl1>0)
    print(rel_error)
    assert rel_error<1e-1

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(num=1, figsize=(8,4))
        plt.subplot(2,3,1);plt.imshow(np.max(lbl1,0));plt.title("GT")
        plt.subplot(2,3,2);plt.imshow(np.max(lbl2,0));plt.title("Reco")
        plt.subplot(2,3,3);plt.imshow(np.max(1*(lbl1>0)+2*(lbl2>0),0));plt.title("Overlay")
        plt.subplot(2,3,4);plt.imshow(np.max(lbl1,1));plt.title("GT")
        plt.subplot(2,3,5);plt.imshow(np.max(lbl2,1));plt.title("Reco")
        plt.subplot(2,3,6);plt.imshow(np.max(1*(lbl1>0)+2*(lbl2>0),1));plt.title("Overlay")
        plt.tight_layout()
        plt.show()
        
    return lbl1, lbl2
    

if __name__ == '__main__':
    lbl1, lbl2 = test_relabel_consistency(128,eps = (.5,1,1.2), plot = True)
    
    # from utils import circle_image

    # rays = Rays_GoldenSpiral(4)
    # lbl = circle_image((64,) * 3)

    # a = star_dist3D(lbl, rays=rays, grid=(1, 2, 2), mode="cpp")
    # b = star_dist3D(lbl, rays=rays, grid=(1, 2, 2), mode="opencl")
    # print(np.amax(np.abs(a - b)))
