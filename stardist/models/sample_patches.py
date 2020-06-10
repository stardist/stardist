"""provides a faster sampling function"""

import numpy as np
from csbdeep.utils import _raise, choice


def sample_patches(datas, patch_size, n_samples, valid_inds=None, verbose=False):
    """optimized version of csbdeep.data.sample_patches_from_multiple_stacks

    samples patch_size from a list of arrays (datas). If the dimension of any array is larger than patch_size, it will broadcast across the last remaining dimensions 
    """

    ndim = len(patch_size)
    dshape = datas[0].shape[:ndim]

    if not all((a.ndim >= ndim for a in datas)):
        raise ValueError("all input array dimension must be have at least %s" % str(len(patch_size)))

    if not all((a.shape[:ndim] == dshape for a in datas)):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,dshape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(dshape)))

    if valid_inds is None:
        valid_inds = tuple(_s.ravel() for _s in np.meshgrid(*tuple(np.arange(p//2,s-p+(p//2)+1) for s,p in zip(dshape, patch_size))))

    n_valid = len(valid_inds[0])

    if n_valid == 0:
        raise ValueError("no regions to sample from!")

    idx = choice(range(n_valid), n_samples, replace=(n_valid < n_samples))
    rand_inds = [v[idx] for v in valid_inds]
    res = [np.stack([data[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,patch_size)) + (slice(None),)*(data.ndim-ndim)] for r in zip(*rand_inds)]) for data in datas]

    return res


def get_valid_inds(x, patch_size, patch_filter = None):
    """
    returns indices of x where patch_filter(x) is true (default: all interior of x)
    """
    len(patch_size)==x.ndim or _raise(ValueError())

    if not all(( 0 < s <= d for s,d in zip(patch_size,x.shape) )):
        raise ValueError("patch_size %s negative or larger than x shape %s along some dimensions" % (str(patch_size), str(x.shape)))

    if patch_filter is None:
        patch_mask = np.ones(x.shape,dtype=np.bool)
    else:
        patch_mask = patch_filter(x, patch_size)

    # get the valid indices

    border_slices = tuple([slice(p // 2, s - p + p // 2 + 1) for p, s in zip(patch_size, x.shape)])
    valid_inds = np.where(patch_mask[border_slices])
    valid_inds = tuple(v + s.start for s, v in zip(border_slices, valid_inds))
    return valid_inds
