from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import numpy as np
import warnings
import os
from zipfile import ZipFile, ZIP_DEFLATED
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage.measurements import find_objects
from skimage.measure import regionprops
from csbdeep.utils import _raise
from csbdeep.utils.six import Path


def path_absolute(path_relative):
    """ Get absolute path to resource"""
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path_relative)


def _is_power_of_2(i):
    assert i > 0
    e = np.log2(i)
    return e == int(e)


def _normalize_grid(grid,n):
    try:
        grid = tuple(grid)
        (len(grid) == n and
         all(map(np.isscalar,grid)) and
         all(map(_is_power_of_2,grid))) or _raise(TypeError())
        return tuple(int(g) for g in grid)
    except (TypeError, AssertionError):
        raise ValueError("grid must be a list/tuple of length {n} with values that are power of 2".format(n=n))


def _check_label_array(y, name=None, check_sequential=False):
    def label_are_sequential(y):
        """ returns true if y has only sequential labels from 1... """
        labels = np.unique(y)
        return (set(labels)-{0}) == set(range(1,1+labels.max()))
    def is_array_of_integers(y):
        """https://stackoverflow.com/a/934652"""
        # return issubclass(y.dtype.type, np.integer)
        return isinstance(y,np.ndarray) and np.issubdtype(y.dtype, np.integer)
    err = ValueError("{label} must be an array of {integers}.".format(
        label = 'labels' if name is None else name,
        integers = ('sequential ' if check_sequential else '') + 'non-negative integers',
    ))
    is_array_of_integers(y) or _raise(err)
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True


def _edt_prob(lbl_img, anisotropy=None):
    if anisotropy is None:
        dist_func = distance_transform_edt
    else:
        from edt import edt as _edt_aniso
        dist_func = lambda img: _edt_aniso(np.ascontiguousarray(img>0), anisotropy=anisotropy)
    prob = np.zeros(lbl_img.shape,np.float32)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        edt = dist_func(mask)[mask]
        prob[mask] = edt/(np.max(edt)+1e-10)
    return prob


def edt_prob(lbl_img, anisotropy=None):
    """Perform EDT on each labeled object and normalize."""
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)

    if anisotropy is None:
        dist_func = distance_transform_edt
    else:
        from edt import edt as _edt_aniso
        dist_func = lambda img: _edt_aniso(np.ascontiguousarray(img>0), anisotropy=anisotropy)

    objects = find_objects(lbl_img)
    prob = np.zeros(lbl_img.shape,np.float32)
    for i,sl in enumerate(objects,1):
        # i: object label id, sl: slices of object in lbl_img
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        # 1. grow object slice by 1 for all interior object bounding boxes
        # 2. perform (correct) EDT for object with label id i
        # 3. extract EDT for object of original slice and normalize
        # 4. store edt for object only for pixels of given label id i
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask = grown_mask[shrink_slice]
        edt = dist_func(grown_mask)[shrink_slice][mask]
        prob[sl][mask] = edt/(np.max(edt)+1e-10)
    return prob


def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def sample_points(n_samples, mask, prob=None, b=2):
    """sample points to draw some of the associated polygons"""
    if b is not None and b > 0:
        # ignore image boundary, since predictions may not be reliable
        mask_b = np.zeros_like(mask)
        mask_b[b:-b,b:-b] = True
    else:
        mask_b = True

    points = np.nonzero(mask & mask_b)

    if prob is not None:
        # weighted sampling via prob
        w = prob[points[0],points[1]].astype(np.float64)
        w /= np.sum(w)
        ind = np.random.choice(len(points[0]), n_samples, replace=True, p=w)
    else:
        ind = np.random.choice(len(points[0]), n_samples, replace=True)

    points = points[0][ind], points[1][ind]
    points = np.stack(points,axis=-1)
    return points


def calculate_extents(lbl, func=np.median):
    if isinstance(lbl,(tuple,list)) or lbl.ndim==4:
        return func(np.stack([calculate_extents(_lbl,func) for _lbl in lbl], axis=0), axis=0)

    lbl.ndim == 3 or _raise(ValueError("label image should be 3 dimensional"))

    regs = regionprops(lbl)
    if len(regs) == 0:
        return np.zeros(3)
    else:
        extents = np.array([np.array(r.bbox[3:])-np.array(r.bbox[:3]) for r in regs])
        return func(extents, axis=0)


def polyroi_bytearray(x,y,pos=None):
    """ Byte array of polygon roi with provided x and y coordinates
        See https://github.com/imagej/imagej1/blob/master/ij/io/RoiDecoder.java
    """
    def _int16(x):
        return int(x).to_bytes(2, byteorder='big', signed=True)
    def _uint16(x):
        return int(x).to_bytes(2, byteorder='big', signed=False)
    def _int32(x):
        return int(x).to_bytes(4, byteorder='big', signed=True)

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    assert len(x) == len(y)
    top, left, bottom, right = y.min(), x.min(), y.max(), x.max() # bbox

    n_coords = len(x)
    bytes_header = 64
    bytes_total = bytes_header + n_coords*2*2
    B = [0] * bytes_total
    B[ 0: 4] = map(ord,'Iout')   # magic start
    B[ 4: 6] = _int16(227)       # version
    B[ 6: 8] = _int16(0)         # roi type (0 = polygon)
    B[ 8:10] = _int16(top)       # bbox top
    B[10:12] = _int16(left)      # bbox left
    B[12:14] = _int16(bottom)    # bbox bottom
    B[14:16] = _int16(right)     # bbox right
    B[16:18] = _uint16(n_coords) # number of coordinates
    if pos is not None:
        B[56:60] = _int32(pos)   # position (C, Z, or T)

    for i,(_x,_y) in enumerate(zip(x,y)):
        xs = bytes_header + 2*i
        ys = xs + 2*n_coords
        B[xs:xs+2] = _int16(_x - left)
        B[ys:ys+2] = _int16(_y - top)

    return bytearray(B)


def export_imagej_rois(fname, polygons, set_position=True, compression=ZIP_DEFLATED):
    """ polygons assumed to be a list/array of arrays with shape (id,x,y) """

    fname = Path(fname)
    if fname.suffix == '.zip':
        fname = Path(fname.stem)

    with ZipFile(str(fname)+'.zip', mode='w', compression=compression) as roizip:
        for pos,polygroup in enumerate(polygons,start=1):
            for i,poly in enumerate(polygroup,start=1):
                roi = polyroi_bytearray(poly[1],poly[0], pos=(pos if set_position else None))
                roizip.writestr('{pos:03d}_{i:03d}.roi'.format(pos=pos,i=i), roi)


# copied from scikit-image master for now (remove when part of a release)
def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.
    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).
    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.
    Returns
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : numpy array of int, shape ``(label_field.max() + 1,)``
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The data type will be the same as `relabeled`.
    inverse_map : 1D numpy array of int, of length offset + number of labels
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The data type will be the same as `relabeled`.
    Notes
    -----
    The label 0 is assumed to denote the background and is never remapped.
    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.
    Examples
    --------
    >>> from skimage.segmentation import relabel_sequential
    >>> label_field = np.array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_sequential(label_field)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> fw
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 5])
    >>> inv
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[label_field] == relab).all()
    True
    >>> (inv[relab] == label_field).all()
    True
    >>> relab, fw, inv = relabel_sequential(label_field, offset=5)
    >>> relab
    array([5, 5, 6, 6, 7, 9, 8])
    """
    offset = int(offset)
    if offset <= 0:
        raise ValueError("Offset must be strictly positive.")
    if np.min(label_field) < 0:
        raise ValueError("Cannot relabel array that contains negative values.")
    m = label_field.max()
    if not np.issubdtype(label_field.dtype, np.integer):
        new_type = np.min_scalar_type(int(m))
        label_field = label_field.astype(new_type)
        m = m.astype(new_type)  # Ensures m is an integer
    labels = np.unique(label_field)
    labels0 = labels[labels != 0]
    required_type = np.min_scalar_type(offset + len(labels0))
    if np.dtype(required_type).itemsize > np.dtype(label_field.dtype).itemsize:
        label_field = label_field.astype(required_type)
    new_labels0 = np.arange(offset, offset + len(labels0))
    if np.all(labels0 == new_labels0):
        return label_field, labels, labels
    forward_map = np.zeros(int(m + 1), dtype=label_field.dtype)
    forward_map[labels0] = new_labels0
    if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
    inverse_map = np.zeros(offset - 1 + len(labels), dtype=label_field.dtype)
    inverse_map[(offset - 1):] = labels
    relabeled = forward_map[label_field]
    return relabeled, forward_map, inverse_map
