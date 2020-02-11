from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from time import time
from .utils import _normalize_grid



def non_maximum_suppression(coord, prob, grid=(1,1), b=2, nms_thresh=0.5, prob_thresh=0.5, verbose=False, max_bbox_search=True):
    """2D coordinates of the polys that survive from a given prediction (prob, coord)

    prob.shape = (Ny,Nx)
    coord.shape = (Ny,Nx,2,n_rays)

    b: don't use pixel closer than b pixels to the image boundary
    """
    from .lib.stardist2d import c_non_max_suppression_inds

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    assert prob.ndim == 2
    assert coord.ndim == 4
    grid = _normalize_grid(grid,2)

    mask = prob > prob_thresh
    if b is not None and b > 0:
        _mask = np.zeros_like(mask)
        _mask[b:-b,b:-b] = True
        mask &= _mask

    polygons = coord[mask]
    scores   = prob[mask]

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    survivors = np.zeros(len(ind), np.bool)
    polygons = polygons[ind]
    scores = scores[ind]

    if max_bbox_search:
        # map pixel indices to ids of sorted polygons (-1 => polygon at that pixel not a candidate)
        mapping = -np.ones(mask.shape,np.int32)
        mapping.flat[ np.flatnonzero(mask)[ind] ] = range(len(ind))
    else:
        mapping = np.empty((0,0),np.int32)

    if verbose:
        t = time()

    survivors[ind] = c_non_max_suppression_inds(polygons.astype(np.int32),
                    mapping, np.float32(nms_thresh), np.int32(max_bbox_search),
                    np.int32(grid[0]), np.int32(grid[1]),np.int32(verbose))

    if verbose:
        print("keeping %s/%s polygons" % (np.count_nonzero(survivors), len(polygons)))
        print("NMS took %.4f s" % (time() - t))

    points = np.stack([ii[survivors] for ii in np.nonzero(mask)],axis=-1)
    return points



def non_maximum_suppression_3d(dist, prob, rays, grid=(1,1,1), b=2, nms_thresh=0.5, prob_thresh=0.5, verbose=False):
    """Non-Maximum-Supression of 3D polyhedra 
    
    Retains only polyhedra whose overlap is smaller than nms_thresh 

    dist.shape = (Nz,Ny,Nx, n_rays)
    prob.shape = (Nz,Ny,Nx)

    """
    
    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    assert prob.ndim == 3 and dist.ndim == 4 and dist.shape[-1] == len(rays) and prob.shape == dist.shape[:3]
    
    grid = _normalize_grid(grid,3)

    verbose and print("predicting instances with prob_thresh = {prob_thresh} and nms_thresh = {nms_thresh}".format(prob_thresh=prob_thresh, nms_thresh=nms_thresh), flush=True)

    ind_thresh = prob > prob_thresh
    if b is not None and b > 0:
        _ind_thresh = np.zeros_like(ind_thresh)
        _ind_thresh[b:-b,b:-b,b:-b] = True
        ind_thresh &= _ind_thresh

    points = np.stack(np.where(ind_thresh), axis=1)
    verbose and print("found %s candidates"%len(points))
    probi = prob[ind_thresh]
    disti = dist[ind_thresh]

    _sorted = np.argsort(probi)[::-1]
    probi = probi[_sorted]
    disti = disti[_sorted]
    points = points[_sorted]

    verbose and print("non-maximum suppression...")
    points = (points * np.array(grid).reshape((1,3)))

    inds = non_maximum_suppression_3d_inds(disti, points, rays=rays, scores=probi, thresh=nms_thresh, verbose=verbose)

    verbose and print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))
    return points[inds], probi[inds], disti[inds]


def non_maximum_suppression_3d_sparse(dist, prob, points, rays, b=2, nms_thresh=0.5, verbose=False):
    """Non-Maximum-Supression of 3D polyhedra from a list of dists, probs and points
    
    Retains only polyhedra whose overlap is smaller than nms_thresh 
    dist.shape = (n_polys, n_rays)
    prob.shape = (n_polys,)
    points.shape = (n_polys,3)
    """
    
    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    dist = np.asarray(dist)
    prob = np.asarray(prob)
    points = np.asarray(points)

    assert dist.ndim == 2 and prob.ndim == 1 and points.ndim == 2 and \
        dist.shape[-1] == len(rays) and points.shape[-1]==3 and len(prob) == len(dist) == len(points)
    
    verbose and print("predicting instances with nms_thresh = {nms_thresh}".format(nms_thresh=nms_thresh), flush=True)

    _sorted = np.argsort(prob)[::-1]
    probi = prob[_sorted]
    disti = dist[_sorted]
    pointsi = points[_sorted]

    verbose and print("non-maximum suppression...")

    inds = non_maximum_suppression_3d_inds(disti, pointsi, rays=rays, scores=probi, thresh=nms_thresh, verbose=verbose)

    verbose and print("keeping %s/%s polyhedra" % (np.count_nonzero(inds), len(inds)))
    return pointsi[inds], probi[inds], disti[inds]


def non_maximum_suppression_3d_inds(dist, points, rays, scores, thresh=0.5, use_bbox=True, verbose=1):
    """
    Applies non maximum supression to ray-convex polyhedra given by dists and rays
    sorted by scores and IoU threshold

    P1 will suppress P2, if IoU(P1,P2) > thresh

    with IoU(P1,P2) = Ainter(P1,P2) / min(A(P1),A(P2))

    i.e. the smaller thresh, the more polygons will be supressed

    dist.shape = (n_poly, n_rays)
    point.shape = (n_poly, 3)
    score.shape = (n_poly,)

    returns indices of selected polygons
    """
    from .lib.stardist3d import c_non_max_suppression_inds

    assert dist.ndim == 2
    assert points.ndim == 2
    assert dist.shape[1] == len(rays)

    n_poly = dist.shape[0]

    if scores is None:
        scores = np.ones(n_poly)

    assert len(scores) == n_poly
    assert points.shape[0] == n_poly

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    survivors = np.ones(n_poly, np.bool)
    dist = dist[ind]
    points = points[ind]
    scores = scores[ind]

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    if verbose:
        t = time()

    survivors[ind] = c_non_max_suppression_inds(_prep(dist, np.float32),
                                                _prep(points, np.float32),
                                                _prep(rays.vertices, np.float32),
                                                _prep(rays.faces, np.int32),
                                                _prep(scores, np.float32),
                                                np.int(use_bbox),
                                                np.int(verbose),
                                                np.float32(thresh))

    if verbose:
        print("NMS took %.4f s" % (time() - t))

    return survivors
