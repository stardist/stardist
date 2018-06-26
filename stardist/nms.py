from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np


def non_maximum_suppression_inds(polygons, scores, thresh=0.5, use_bbox=True):
    """
    Applies non maximum supression to given (ray-shaped) polygons, scores and IoU threshold

    polygons.shape = (n_polygons, 2, n_rays)
    score.shape = (n_polygons,)

    returns indices of selected polygons
    """

    from .lib.stardist import c_non_max_suppression_inds
    assert len(polygons) == len(scores)

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    survivors = np.zeros(len(ind), np.bool)
    polygons = polygons[ind]
    scores = scores[ind]

    survivors[ind] = c_non_max_suppression_inds(polygons.astype(np.int32), scores.astype(np.float32), np.float32(thresh), np.bool(use_bbox))
    return survivors


def non_maximum_suppression(coord, prob, b=2, nms_thresh=0.5, prob_thresh=0.5, verbose=False):
    """2D coordinates of the polys that survive from a given prediction (prob, coord)

    prob.shape = (Ny,Nx)
    coord.shape = (Ny,Nx,2,n_rays)

    b: don't use pixel closer than b pixels to the image boundary
    """

    assert prob.ndim == 2
    assert coord.ndim == 4

    mask = prob > prob_thresh
    if b is not None and b > 0:
        _mask = np.zeros_like(mask)
        _mask[b:-b,b:-b] = True
        mask &= _mask

    polygons = coord[mask]
    scores   = prob[mask]

    survivors = non_maximum_suppression_inds(polygons, scores, thresh=nms_thresh)

    if verbose:
        print("keeping %s/%s polygons" % (np.count_nonzero(survivors), len(polygons)))

    points = np.stack([ii[survivors] for ii in np.nonzero(mask)],axis=-1)
    return points
