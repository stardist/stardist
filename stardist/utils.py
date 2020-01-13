from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import warnings
import os
import datetime
from tqdm import tqdm
from zipfile import ZipFile, ZIP_DEFLATED
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage.measurements import find_objects
from scipy.optimize import minimize_scalar
from skimage.measure import regionprops
from csbdeep.utils import _raise
from csbdeep.utils.six import Path

from .matching import matching_dataset


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


def _edt_dist_func(anisotropy):
    try:
        from edt import edt as edt_func
        # raise ImportError()
        dist_func = lambda img: edt_func(np.ascontiguousarray(img>0), anisotropy=anisotropy)
    except ImportError:
        dist_func = lambda img: distance_transform_edt(img, sampling=anisotropy)
    return dist_func


def _edt_prob(lbl_img, anisotropy=None):
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        lbl_img = np.pad(lbl_img, ((1,1),)*lbl_img.ndim, mode='constant')
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")
    dist_func = _edt_dist_func(anisotropy)
    prob = np.zeros(lbl_img.shape,np.float32)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        edt = dist_func(mask)[mask]
        prob[mask] = edt/(np.max(edt)+1e-10)
    if constant_img:
        prob = prob[(slice(1,-1),)*lbl_img.ndim].copy()
    return prob


def edt_prob(lbl_img, anisotropy=None):
    """Perform EDT on each labeled object and normalize."""
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        lbl_img = np.pad(lbl_img, ((1,1),)*lbl_img.ndim, mode='constant')
        warnings.warn("EDT of constant label image is ill-defined. (Assuming background around it.)")
    dist_func = _edt_dist_func(anisotropy)
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
    if isinstance(lbl,(tuple,list)) or (isinstance(lbl,np.ndarray) and lbl.ndim==4):
        return func(np.stack([calculate_extents(_lbl,func) for _lbl in lbl], axis=0), axis=0)

    n = lbl.ndim
    n in (2,3) or _raise(ValueError("label image should be 2- or 3-dimensional (or pass a list of these)"))

    regs = regionprops(lbl)
    if len(regs) == 0:
        return np.zeros(n)
    else:
        extents = np.array([np.array(r.bbox[n:])-np.array(r.bbox[:n]) for r in regs])
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
    x = np.round(x)
    y = np.round(y)
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
    """ polygons assumed to be a list of arrays with shape (id,2,c) """

    if isinstance(polygons,np.ndarray):
        polygons = (polygons,)

    fname = Path(fname)
    if fname.suffix == '.zip':
        fname = fname.with_suffix('')

    with ZipFile(str(fname)+'.zip', mode='w', compression=compression) as roizip:
        for pos,polygroup in enumerate(polygons,start=1):
            for i,poly in enumerate(polygroup,start=1):
                roi = polyroi_bytearray(poly[1],poly[0], pos=(pos if set_position else None))
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
