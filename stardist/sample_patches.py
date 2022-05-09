"""provides a faster sampling function"""

import numpy as np
from csbdeep.utils import _raise, choice


def sample_patches(datas, patch_size, n_samples, valid_inds=None, verbose=False):
    """optimized version of csbdeep.data.sample_patches_from_multiple_stacks
    """

    len(patch_size)==datas[0].ndim or _raise(ValueError())

    if not all(( a.shape == datas[0].shape for a in datas )):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,datas[0].shape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(datas[0].shape)))

    if valid_inds is None:
        valid_inds = tuple(_s.ravel() for _s in np.meshgrid(*tuple(np.arange(p//2,s-p//2+1) for s,p in zip(datas[0].shape, patch_size))))

    n_valid = len(valid_inds[0])

    if n_valid == 0:
        raise ValueError("no regions to sample from!")

    idx = choice(range(n_valid), n_samples, replace=(n_valid < n_samples))
    rand_inds = [v[idx] for v in valid_inds]
    res = [np.stack([data[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,patch_size))] for r in zip(*rand_inds)]) for data in datas]

    return res


def get_valid_inds(img, patch_size, patch_filter=None):
    """
    Returns all indices of an image that 
    - can be used as center points for sampling patches of a given patch_size, and
    - are part of the boolean mask given by the function patch_filter (if provided)

    img: np.ndarray
    patch_size: tuple of ints 
        the width of patches per img dimension, 
    patch_filter: None or callable
        a function with signature patch_filter(img, patch_size) returning a boolean mask 
    """

    len(patch_size)==img.ndim or _raise(ValueError())

    if not all(( 0 < s <= d for s,d in zip(patch_size,img.shape))):
        raise ValueError("patch_size %s negative or larger than image shape %s along some dimensions" % (str(patch_size), str(img.shape)))

    if patch_filter is None:
        # only cut border indices (which is faster)
        patch_mask = np.ones(img.shape,dtype=bool)
        valid_inds = tuple(np.arange(p // 2, s - p + p // 2 + 1).astype(np.uint32) for p, s in zip(patch_size, img.shape))
        valid_inds = tuple(s.ravel() for s in np.meshgrid(*valid_inds, indexing='ij'))
    else:
        patch_mask = patch_filter(img, patch_size)

        # get the valid indices
        border_slices = tuple([slice(p // 2, s - p + p // 2 + 1) for p, s in zip(patch_size, img.shape)])
        valid_inds = np.where(patch_mask[border_slices])
        valid_inds = tuple((v + s.start).astype(np.uint32) for s, v in zip(border_slices, valid_inds))

    return valid_inds
