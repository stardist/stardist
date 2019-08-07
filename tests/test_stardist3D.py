import numpy as np
import pytest
from stardist import star_dist3D, Rays_GoldenSpiral
from utils import random_image, real_image3d, check_similar


@pytest.mark.parametrize('img', (real_image3d(), random_image((33, 44, 55))))
@pytest.mark.parametrize('n_rays', (4, 16, 32))
@pytest.mark.parametrize('grid', ((1,1,1),(1,2,4)))
def test_types(img, n_rays, grid):
    mode = "cpp"
    rays = Rays_GoldenSpiral(n_rays)
    gt = star_dist3D(img, rays=rays, grid=grid, mode=mode)
    for dtype in (np.int8, np.int16, np.int32,
                  np.uint8, np.uint16, np.uint32):
        x = star_dist3D(img.astype(dtype), rays=rays, grid=grid, mode=mode)
        print(f"test_stardist3D (mode {mode}) for shape {img.shape} and type {dtype}")
        check_similar(gt, x)


@pytest.mark.gpu
@pytest.mark.parametrize('img', (real_image3d(), random_image((33, 44, 55))))
@pytest.mark.parametrize('n_rays', (4, 16, 32))
@pytest.mark.parametrize('grid', ((1,1,1),(1,2,4)))
def test_types_gpu(img, n_rays, grid):
    mode = "opencl"
    rays = Rays_GoldenSpiral(n_rays)
    gt = star_dist3D(img, rays=rays, grid=grid, mode=mode)
    for dtype in (np.int8, np.int16, np.int32,
                  np.uint8, np.uint16, np.uint32):
        x = star_dist3D(img.astype(dtype), rays=rays, grid=grid, mode=mode)
        print(f"test_stardist3D (mode {mode}) for shape {img.shape} and type {dtype}")
        check_similar(gt, x)


@pytest.mark.gpu
@pytest.mark.parametrize('img', (real_image3d(), random_image((33, 44, 55))))
@pytest.mark.parametrize('n_rays', (4, 16, 32))
@pytest.mark.parametrize('grid', ((1,1,1),(1,2,4)))
def test_cpu_gpu(img, n_rays, grid):
    rays = Rays_GoldenSpiral(n_rays)
    s_cpp = star_dist3D(img, rays=rays, grid=grid, mode="cpp")
    s_ocl = star_dist3D(img, rays=rays, grid=grid, mode="opencl")
    check_similar(s_cpp, s_ocl)


if __name__ == '__main__':
    from utils import circle_image

    rays = Rays_GoldenSpiral(4)
    lbl = circle_image((64,) * 3)

    a = star_dist3D(lbl, rays=rays, grid=(1, 2, 2), mode="cpp")
    b = star_dist3D(lbl, rays=rays, grid=(1, 2, 2), mode="opencl")
    print(np.amax(np.abs(a - b)))
