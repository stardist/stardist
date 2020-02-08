from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import warnings

from skimage.measure import regionprops
from skimage.draw import polygon
from csbdeep.utils import _raise

from ..utils import path_absolute, _is_power_of_2, _normalize_grid
from ..matching import _check_label_array
from ..lib.stardist2d import c_star_dist



def _ocl_star_dist(a, n_rays=32):
    from gputools import OCLProgram, OCLArray, OCLImage
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
    n_rays = int(n_rays)
    src = OCLImage.from_array(a.astype(np.uint16,copy=False))
    dst = OCLArray.empty(a.shape+(n_rays,), dtype=np.float32)
    program = OCLProgram(path_absolute("kernels/stardist2d.cl"), build_options=['-D', 'N_RAYS=%d' % n_rays])
    program.run_kernel('star_dist', src.shape, None, dst.data, src)
    return dst.get()


def _cpp_star_dist(a, n_rays=32):
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
    return c_star_dist(a.astype(np.uint16,copy=False), int(n_rays))


def _py_star_dist(a, n_rays=32):
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
    n_rays = int(n_rays)
    a = a.astype(np.uint16,copy=False)
    dst = np.empty(a.shape+(n_rays,),np.float32)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            value = a[i,j]
            if value == 0:
                dst[i,j] = 0
            else:
                st_rays = np.float32((2*np.pi) / n_rays)
                for k in range(n_rays):
                    phi = np.float32(k*st_rays)
                    dy = np.cos(phi)
                    dx = np.sin(phi)
                    x, y = np.float32(0), np.float32(0)
                    while True:
                        x += dx
                        y += dy
                        ii = int(round(i+x))
                        jj = int(round(j+y))
                        if (ii < 0 or ii >= a.shape[0] or
                            jj < 0 or jj >= a.shape[1] or
                            value != a[ii,jj]):
                            # small correction as we overshoot the boundary
                            t_corr = 1-.5/max(np.abs(dx),np.abs(dy))
                            x -= t_corr*dx
                            y -= t_corr*dy
                            dist = np.sqrt(x**2+y**2)
                            dst[i,j,k] = dist
                            break
    return dst


def star_dist(a, n_rays=32, mode='cpp'):
    """'a' assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""

    n_rays >= 3 or _raise(ValueError("need 'n_rays' >= 3"))

    if mode == 'python':
        return _py_star_dist(a, n_rays)
    elif mode == 'cpp':
        return _cpp_star_dist(a, n_rays)
    elif mode == 'opencl':
        return _ocl_star_dist(a, n_rays)
    else:
        _raise(ValueError("Unknown mode %s" % mode))


def polygons_to_label(coord, prob, points, shape=None, thr=-np.inf):
    sh = coord.shape[:2] if shape is None else shape
    lbl = np.zeros(sh,np.int32)
    # sort points with increasing probability
    ind = np.argsort([ prob[p[0],p[1]] for p in points ])
    points = points[ind]

    i = 1
    for p in points:
        if prob[p[0],p[1]] < thr:
            continue
        rr,cc = polygon(coord[p[0],p[1],0], coord[p[0],p[1],1], sh)
        lbl[rr,cc] = i
        i += 1

    return lbl


def relabel_image_stardist(lbl, n_rays, **kwargs):
    """relabel each label region in `lbl` with its star representation"""
    _check_label_array(lbl, "lbl")
    dist = star_dist(lbl, n_rays, **kwargs)
    coord = dist_to_coord(dist)
    points = np.array(tuple(np.array(r.centroid).astype(int) for r in regionprops(lbl)))
    return polygons_to_label(coord, np.ones_like(lbl), points, shape=lbl.shape)


def ray_angles(n_rays=32):
    return np.linspace(0,2*np.pi,n_rays,endpoint=False)


def dist_to_coord(rhos, grid=(1,1)):
    """convert from polar to cartesian coordinates for a single image (3-D array) or multiple images (4-D array)"""

    grid = _normalize_grid(grid,2)
    is_single_image = rhos.ndim == 3
    if is_single_image:
        rhos = np.expand_dims(rhos,0)
    assert rhos.ndim == 4

    n_images,h,w,n_rays = rhos.shape
    coord = np.empty((n_images,h,w,2,n_rays),dtype=rhos.dtype)

    start = np.indices((h,w))
    for i in range(2):
        coord[...,i,:] = grid[i] * np.broadcast_to(start[i].reshape(1,h,w,1), (n_images,h,w,n_rays))

    phis = ray_angles(n_rays).reshape(1,1,1,n_rays)

    coord[...,0,:] += rhos * np.sin(phis) # row coordinate
    coord[...,1,:] += rhos * np.cos(phis) # col coordinate

    return coord[0] if is_single_image else coord
