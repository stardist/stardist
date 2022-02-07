from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import warnings

from skimage.measure import regionprops
from skimage.draw import polygon
from csbdeep.utils import _raise

from ..utils import path_absolute, _is_power_of_2, _normalize_grid
from ..matching import _check_label_array
from ..lib.stardist2d import c_star_dist



def _ocl_star_dist(lbl, n_rays=32, grid=(1,1)):
    from gputools import OCLProgram, OCLArray, OCLImage
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
    n_rays = int(n_rays)
    # slicing with grid is done with tuple(slice(0, None, g) for g in grid)
    res_shape = tuple((s-1)//g+1 for s, g in zip(lbl.shape, grid))
    
    src = OCLImage.from_array(lbl.astype(np.uint16,copy=False))
    dst = OCLArray.empty(res_shape+(n_rays,), dtype=np.float32)
    program = OCLProgram(path_absolute("kernels/stardist2d.cl"), build_options=['-D', 'N_RAYS=%d' % n_rays])
    program.run_kernel('star_dist', res_shape[::-1], None, dst.data, src, np.int32(grid[0]),np.int32(grid[1]))
    return dst.get()


def _cpp_star_dist(lbl, n_rays=32, grid=(1,1)):
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
    return c_star_dist(lbl.astype(np.uint16,copy=False), np.int32(n_rays), np.int32(grid[0]),np.int32(grid[1]))


def _py_star_dist(a, n_rays=32, grid=(1,1)):
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
    if grid != (1,1):
        raise NotImplementedError(grid)
    
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


def star_dist(a, n_rays=32, grid=(1,1), mode='cpp'):
    """'a' assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""

    n_rays >= 3 or _raise(ValueError("need 'n_rays' >= 3"))

    if mode == 'python':
        return _py_star_dist(a, n_rays, grid=grid)
    elif mode == 'cpp':
        return _cpp_star_dist(a, n_rays, grid=grid)
    elif mode == 'opencl':
        return _ocl_star_dist(a, n_rays, grid=grid)
    else:
        _raise(ValueError("Unknown mode %s" % mode))


def _dist_to_coord_old(rhos, grid=(1,1)):
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


def _polygons_to_label_old(coord, prob, points, shape=None, thr=-np.inf):
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


def dist_to_coord(dist, points, scale_dist=(1,1)):
    """convert from polar to cartesian coordinates for a list of distances and center points
    dist.shape   = (n_polys, n_rays)
    points.shape = (n_polys, 2)
    len(scale_dist) = 2
    return coord.shape = (n_polys,2,n_rays)
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    assert dist.ndim==2 and points.ndim==2 and len(dist)==len(points) \
        and points.shape[1]==2 and len(scale_dist)==2
    n_rays = dist.shape[1]
    phis = ray_angles(n_rays)
    coord = (dist[:,np.newaxis]*np.array([np.sin(phis),np.cos(phis)])).astype(np.float32)
    coord *= np.asarray(scale_dist).reshape(1,2,1)    
    coord += points[...,np.newaxis] 
    return coord


def polygons_to_label_coord(coord, shape, labels=None):
    """renders polygons to image of given shape

    coord.shape   = (n_polys, n_rays)
    """
    coord = np.asarray(coord)
    if labels is None: labels = np.arange(len(coord))

    _check_label_array(labels, "labels")
    assert coord.ndim==3 and coord.shape[1]==2 and len(coord)==len(labels)

    lbl = np.zeros(shape,np.int32)

    for i,c in zip(labels,coord):
        rr,cc = polygon(*c, shape)
        lbl[rr,cc] = i+1

    return lbl


def polygons_to_label(dist, points, shape, prob=None, thr=-np.inf, scale_dist=(1,1)):
    """converts distances and center points to label image

    dist.shape   = (n_polys, n_rays)
    points.shape = (n_polys, 2)

    label ids will be consecutive and adhere to the order given
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    prob = np.inf*np.ones(len(points)) if prob is None else np.asarray(prob)

    assert dist.ndim==2 and points.ndim==2 and len(dist)==len(points)
    assert len(points)==len(prob) and points.shape[1]==2 and prob.ndim==1

    n_rays = dist.shape[1]

    ind = prob>thr
    points = points[ind]
    dist = dist[ind]
    prob = prob[ind]

    ind = np.argsort(prob, kind='stable')
    points = points[ind]
    dist = dist[ind]

    coord = dist_to_coord(dist, points, scale_dist=scale_dist)

    return polygons_to_label_coord(coord, shape=shape, labels=ind)


def relabel_image_stardist(lbl, n_rays, **kwargs):
    """relabel each label region in `lbl` with its star representation"""
    _check_label_array(lbl, "lbl")
    if not lbl.ndim==2:
        raise ValueError("lbl image should be 2 dimensional")
    dist = star_dist(lbl, n_rays, **kwargs)
    points = np.array(tuple(np.array(r.centroid).astype(int) for r in regionprops(lbl)))
    dist = dist[tuple(points.T)]
    return polygons_to_label(dist, points, shape=lbl.shape)


def ray_angles(n_rays=32):
    return np.linspace(0,2*np.pi,n_rays,endpoint=False)
