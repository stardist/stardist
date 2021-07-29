from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import warnings
import os
import datetime
from tqdm import tqdm
from collections import defaultdict
from zipfile import ZipFile, ZIP_DEFLATED
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage.measurements import find_objects
from scipy.optimize import minimize_scalar
from skimage.measure import regionprops
from csbdeep.utils import _raise
from csbdeep.utils.six import Path
from collections.abc import Iterable

from .matching import matching_dataset, _check_label_array


try:
    from edt import edt
    _edt_available = True
    try:    _edt_parallel_max = len(os.sched_getaffinity(0))
    except: _edt_parallel_max = 128
    _edt_parallel_default = 4
    _edt_parallel = os.environ.get('STARDIST_EDT_NUM_THREADS', _edt_parallel_default)
    try:
        _edt_parallel = min(_edt_parallel_max, int(_edt_parallel))
    except ValueError as e:
        warnings.warn(f"Invalid value ({_edt_parallel}) for STARDIST_EDT_NUM_THREADS. Using default value ({_edt_parallel_default}) instead.")
        _edt_parallel = _edt_parallel_default
    del _edt_parallel_default, _edt_parallel_max
except ImportError:
    _edt_available = False
    # warnings.warn("Could not find package edt... \nConsider installing it with \n  pip install edt\nto improve training data generation performance.")
    pass


def gputools_available():
    try:
        import gputools
    except:
        return False
    return True


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
        raise ValueError("grid = {grid} must be a list/tuple of length {n} with values that are power of 2".format(grid=grid, n=n))


def edt_prob(lbl_img, anisotropy=None):
    if _edt_available:
        return _edt_prob_edt(lbl_img, anisotropy=anisotropy)
    else:
        # warnings.warn("Could not find package edt... \nConsider installing it with \n  pip install edt\nto improve training data generation performance.")
        return _edt_prob_scipy(lbl_img, anisotropy=anisotropy)

def _edt_prob_edt(lbl_img, anisotropy=None):
    """Perform EDT on each labeled object and normalize.
    Internally uses https://github.com/seung-lab/euclidean-distance-transform-3d
    that can handle multiple labels at once
    """
    lbl_img = np.ascontiguousarray(lbl_img)
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")
    # we just need to compute the edt once but then normalize it for each object
    prob = edt(lbl_img, anisotropy=anisotropy, black_border=constant_img, parallel=_edt_parallel)
    objects = find_objects(lbl_img)
    for i,sl in enumerate(objects,1):
        # i: object label id, sl: slices of object in lbl_img
        if sl is None: continue
        _mask = lbl_img[sl]==i
        # normalize it
        prob[sl][_mask] /= np.max(prob[sl][_mask]+1e-10)
    return prob

def _edt_prob_scipy(lbl_img, anisotropy=None):
    """Perform EDT on each labeled object and normalize."""
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        lbl_img = np.pad(lbl_img, ((1,1),)*lbl_img.ndim, mode='constant')
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")
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
        edt = distance_transform_edt(grown_mask, sampling=anisotropy)[shrink_slice][mask]
        prob[sl][mask] = edt/(np.max(edt)+1e-10)
    if constant_img:
        prob = prob[(slice(1,-1),)*lbl_img.ndim].copy()
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
    """ Aggregate bounding box sizes of objects in label images. """
    if (isinstance(lbl,np.ndarray) and lbl.ndim==4) or (not isinstance(lbl,np.ndarray) and  isinstance(lbl,Iterable)):
        return func(np.stack([calculate_extents(_lbl,func) for _lbl in lbl], axis=0), axis=0)

    n = lbl.ndim
    n in (2,3) or _raise(ValueError("label image should be 2- or 3-dimensional (or pass a list of these)"))

    regs = regionprops(lbl)
    if len(regs) == 0:
        return np.zeros(n)
    else:
        extents = np.array([np.array(r.bbox[n:])-np.array(r.bbox[:n]) for r in regs])
        return func(extents, axis=0)


def polyroi_bytearray(x,y,pos=None,subpixel=True):
    """ Byte array of polygon roi with provided x and y coordinates
        See https://github.com/imagej/imagej1/blob/master/ij/io/RoiDecoder.java
    """
    import struct
    def _int16(x):
        return int(x).to_bytes(2, byteorder='big', signed=True)
    def _uint16(x):
        return int(x).to_bytes(2, byteorder='big', signed=False)
    def _int32(x):
        return int(x).to_bytes(4, byteorder='big', signed=True)
    def _float(x):
        return struct.pack(">f", x)

    subpixel = bool(subpixel)
    # add offset since pixel center is at (0.5,0.5) in ImageJ
    x_raw = np.asarray(x).ravel() + 0.5
    y_raw = np.asarray(y).ravel() + 0.5
    x = np.round(x_raw)
    y = np.round(y_raw)
    assert len(x) == len(y)
    top, left, bottom, right = y.min(), x.min(), y.max(), x.max() # bbox

    n_coords = len(x)
    bytes_header = 64
    bytes_total = bytes_header + n_coords*2*2 + subpixel*n_coords*2*4
    B = [0] * bytes_total
    B[ 0: 4] = map(ord,'Iout')   # magic start
    B[ 4: 6] = _int16(227)       # version
    B[ 6: 8] = _int16(0)         # roi type (0 = polygon)
    B[ 8:10] = _int16(top)       # bbox top
    B[10:12] = _int16(left)      # bbox left
    B[12:14] = _int16(bottom)    # bbox bottom
    B[14:16] = _int16(right)     # bbox right
    B[16:18] = _uint16(n_coords) # number of coordinates
    if subpixel:
        B[50:52] = _int16(128)   # subpixel resolution (option flag)
    if pos is not None:
        B[56:60] = _int32(pos)   # position (C, Z, or T)

    for i,(_x,_y) in enumerate(zip(x,y)):
        xs = bytes_header + 2*i
        ys = xs + 2*n_coords
        B[xs:xs+2] = _int16(_x - left)
        B[ys:ys+2] = _int16(_y - top)

    if subpixel:
        base1 = bytes_header + n_coords*2*2
        base2 = base1 + n_coords*4
        for i,(_x,_y) in enumerate(zip(x_raw,y_raw)):
            xs = base1 + 4*i
            ys = base2 + 4*i
            B[xs:xs+4] = _float(_x)
            B[ys:ys+4] = _float(_y)

    return bytearray(B)


def export_imagej_rois(fname, polygons, set_position=True, subpixel=True, compression=ZIP_DEFLATED):
    """ polygons assumed to be a list of arrays with shape (id,2,c) """

    if isinstance(polygons,np.ndarray):
        polygons = (polygons,)

    fname = Path(fname)
    if fname.suffix == '.zip':
        fname = fname.with_suffix('')

    with ZipFile(str(fname)+'.zip', mode='w', compression=compression) as roizip:
        for pos,polygroup in enumerate(polygons,start=1):
            for i,poly in enumerate(polygroup,start=1):
                roi = polyroi_bytearray(poly[1],poly[0], pos=(pos if set_position else None), subpixel=subpixel)
                roizip.writestr('{pos:03d}_{i:03d}.roi'.format(pos=pos,i=i), roi)


def optimize_threshold(Y, Yhat, model, nms_thresh, measure='accuracy', iou_threshs=[0.3,0.5,0.7], bracket=None, tol=1e-2, maxiter=20, verbose=1):
    """ Tune prob_thresh for provided (fixed) nms_thresh to maximize matching score (for given measure and averaged over iou_threshs). """
    np.isscalar(nms_thresh) or _raise(ValueError("nms_thresh must be a scalar"))
    iou_threshs = [iou_threshs] if np.isscalar(iou_threshs) else iou_threshs
    values = dict()

    if bracket is None:
        max_prob = max([np.max(prob) for prob, dist in Yhat])
        bracket = max_prob/2, max_prob
    # print("bracket =", bracket)

    with tqdm(total=maxiter, disable=(verbose!=1), desc="NMS threshold = %g" % nms_thresh) as progress:

        def fn(thr):
            prob_thresh = np.clip(thr, *bracket)
            value = values.get(prob_thresh)
            if value is None:
                Y_instances = [model._instances_from_prediction(y.shape, *prob_dist, prob_thresh=prob_thresh, nms_thresh=nms_thresh)[0] for y,prob_dist in zip(Y,Yhat)]
                stats = matching_dataset(Y, Y_instances, thresh=iou_threshs, show_progress=False, parallel=True)
                values[prob_thresh] = value = np.mean([s._asdict()[measure] for s in stats])
            if verbose > 1:
                print("{now}   thresh: {prob_thresh:f}   {measure}: {value:f}".format(
                    now = datetime.datetime.now().strftime('%H:%M:%S'),
                    prob_thresh = prob_thresh,
                    measure = measure,
                    value = value,
                ), flush=True)
            else:
                progress.update()
                progress.set_postfix_str("{prob_thresh:.3f} -> {value:.3f}".format(prob_thresh=prob_thresh, value=value))
                progress.refresh()
            return -value

        opt = minimize_scalar(fn, method='golden', bracket=bracket, tol=tol, options={'maxiter': maxiter})

    verbose > 1 and print('\n',opt, flush=True)
    return opt.x, -opt.fun


def _invert_dict(d):
    """ return  v-> [k_1,k_2,k_3....] for k,v in d"""
    res = defaultdict(list)
    for k,v in d.items():
        res[v].append(k)
    return res


def mask_to_categorical(y, n_classes, classes, return_cls_dict=False):
    """generates a multi-channel categorical class map

    Parameters
    ----------
    y : n-dimensional ndarray
        integer label array
    n_classes : int
        Number of different classes (without background)
    classes: dict, integer, or None
        the label to class assignment
        can be
        - dict {label -> class_id}
           the value of class_id can be
                             0   -> background class
                  1...n_classes  -> the respective object class (1 ... n_classes)
                           None  -> ignore object (prob is set to -1 for the pixels of the object, except for background class)
        - single integer value or None -> broadcast value to all labels

    Returns
    -------
    probability map of shape y.shape+(n_classes+1,) (first channel is background)

    """

    _check_label_array(y, 'y')
    if not (np.issubdtype(type(n_classes), np.integer) and n_classes>=1):
        raise ValueError(f"n_classes is '{n_classes}' but should be a positive integer")

    y_labels = np.unique(y[y>0]).tolist()

    # build dict class_id -> labels (inverse of classes)
    if np.issubdtype(type(classes), np.integer) or classes is None:
        classes = dict((k,classes) for k in y_labels)
    elif isinstance(classes, dict):
        pass
    else:
        raise ValueError("classes should be dict, single scalar, or None!")

    if not set(y_labels).issubset(set(classes.keys())):
        raise ValueError(f"all gt labels should be present in class dict provided \ngt_labels found\n{set(y_labels)}\nclass dict labels provided\n{set(classes.keys())}")

    cls_dict = _invert_dict(classes)

    # prob map
    y_mask = np.zeros(y.shape+(n_classes+1,), np.float32)

    for cls, labels in cls_dict.items():
        if cls is None:
            # prob == -1 will be used in the loss to ignore object
            y_mask[np.isin(y, labels)] = -1
        elif np.issubdtype(type(cls), np.integer) and 0 <= cls <= n_classes:
            y_mask[...,cls] = np.isin(y, labels)
        else:
            raise ValueError(f"Wrong class id '{cls}' (for n_classes={n_classes})")

    # set 0/1 background prob (unaffected by None values for class ids)
    y_mask[...,0] = (y==0)

    if return_cls_dict:
        return y_mask, cls_dict
    else:
        return y_mask


def _is_floatarray(x):
    return isinstance(x.dtype.type(0),np.floating)


def abspath(root, relpath):
    from pathlib import Path
    root = Path(root)
    if root.is_dir():
        path = root/relpath
    else:
        path = root.parent/relpath
    return str(path.absolute())
