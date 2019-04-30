from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np



def non_maximum_suppression(coord, prob, b=2, nms_thresh=0.5, prob_thresh=0.5, verbose=False, max_bbox_search=True):
    """2D coordinates of the polys that survive from a given prediction (prob, coord)

    prob.shape = (Ny,Nx)
    coord.shape = (Ny,Nx,2,n_rays)

    b: don't use pixel closer than b pixels to the image boundary
    """
    from .lib.stardist import c_non_max_suppression_inds

    assert prob.ndim == 2
    assert coord.ndim == 4

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

    survivors[ind] = c_non_max_suppression_inds(polygons.astype(np.int32), mapping, np.float32(nms_thresh), np.int32(max_bbox_search))

    if verbose:
        print("keeping %s/%s polygons" % (np.count_nonzero(survivors), len(polygons)))

    points = np.stack([ii[survivors] for ii in np.nonzero(mask)],axis=-1)
    return points
