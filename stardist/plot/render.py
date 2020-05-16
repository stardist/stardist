from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from csbdeep.utils import normalize
from ..matching import matching
from .plot import random_label_cmap


def _single_color_integer_cmap(color = (.3,.4,.5)):
    from matplotlib.colors import Colormap
    
    assert len(color) in (3,4)
    
    class BinaryMap(Colormap):
        def __init__(self, color):
            self.color = np.array(color)
            if len(self.color)==3:
                self.color = np.concatenate([self.color,[1]])
        def __call__(self, X, alpha=None, bytes=False):
            res = np.zeros(X.shape+(4,), np.float32)
            res[...,-1] = self.color[-1]
            res[X>0] = np.expand_dims(self.color,0)
            if bytes:
                return np.clip(256*res,0,255).astype(np.uint8)
            else:
                return res
    return BinaryMap(color)
                       


def plot_linearmap(cdict):
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)


def render_label(lbl, img = None, cmap = None, cmap_img = "gray", alpha = 0.5, alpha_boundary = None, normalize_img = True):
    """Renders a label image and optionally overlays it with another image. Used for generating simple output images to asses the label quality

    Parameters
    ----------
    lbl: np.ndarray of dtype np.uint16
        The 2D label image 
    img: np.ndarray 
        The array to overlay the label image with (optional)
    cmap: string, tuple, or callable
        The label colormap. If given as rgb(a)  only a single color is used, if None uses a random colormap 
    cmap_img: string or callable
        The colormap of img (optional)
    alpha: float 
        The alpha value of the overlay. Set alpha=1 to get fully opaque labels
    alpha_boundary: float
        The alpha value of the boundary (if None, use the same as for labels, i.e. no boundaries are visible)
    normalize_img: bool
        If True, normalizes the img (if given)

    Example
    ======= 

    from scipy.ndimage import label, zoom      
    img = zoom(np.random.uniform(0,1,(16,16)),(8,8),order=3)            
    lbl,_ = label(img>.8)
    u1 = render_label(lbl, img = img, alpha = .7)
    u2 = render_label(lbl, img = img, alpha = 0, alpha_boundary =.8)
    plt.subplot(1,2,1);plt.imshow(u1)
    plt.subplot(1,2,2);plt.imshow(u2)

    """
    from skimage.segmentation import find_boundaries
    from matplotlib import cm
    
    alpha = np.clip(alpha, 0, 1)

    if alpha_boundary is None:
        alpha_boundary = alpha
        
    if cmap is None:
        cmap = random_label_cmap()
    elif isinstance(cmap, tuple):
        cmap = _single_color_integer_cmap(cmap)
    else:
        pass
        
    cmap = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cmap_img = cm.get_cmap(cmap_img) if isinstance(cmap_img, str) else cmap_img

    # render image if given
    if img is None:
        im_img = np.zeros(lbl.shape+(4,),np.float32)
        im_img[...,-1] = 1
        
    else:
        assert lbl.shape[:2] == img.shape[:2]
        img = normalize(img) if normalize_img else img
        if img.ndim==2:
            im_img = cmap_img(img)
        elif img.ndim==3:
            im_img = img[...,:4]
            if img.shape[-1]<4:
                im_img = np.concatenate([img, np.ones(img.shape[:2]+(4-img.shape[-1],))], axis = -1)
        else:
            raise ValueError("img should be 2 or 3 dimensional")
            
                
            
    # render label
    im_lbl = cmap(lbl)

    mask_lbl = lbl>0
    mask_bound = np.bitwise_and(mask_lbl,find_boundaries(lbl, mode = "thick"))
    
    # blend
    im = im_img.copy()
    
    im[mask_lbl] = alpha*im_lbl[mask_lbl]+(1-alpha)*im_img[mask_lbl]
    im[mask_bound] = alpha_boundary*im_lbl[mask_bound]+(1-alpha_boundary)*im_img[mask_bound]
        
    return im


def random_hls(n=2**16, h0 = .33, l0 = (.8,1), s0 = (.5,.8)  ):
    """
    h0 = 0 -> red
    h0 = 0.33 -> green
    h0 = 0.66 -> blue
    h0 = 0.833 -> magenta
    """
    _f = lambda x: (x,)*2 if np.isscalar(x) else tuple(x)
    h0,s0,l0 = map(_f,(h0,s0,l0))

    h = np.random.uniform(*h0,n)
    s = np.random.uniform(*s0,n)
    l = np.random.uniform(*l0,n)

    return h,l,s

def cmap_from_hls(h,l,s):
    import matplotlib
    import colorsys
    h = h%1
    l = np.clip(l,0,1)
    s = np.clip(s,0,1)

    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

def match_labels(y0,y):
    """match labels from y to y0"""
    res = matching(y0,y,report_matches=True, thresh =.1)
    if len(res.matched_pairs)==0:
        print("no matching found")
        return y
    
    ind_matched0, ind_matched = tuple(zip(*res.matched_pairs))
    
    ind_unmatched = (set(np.unique(y))-{0}) - set(ind_matched)

    leftover_labels = set(np.arange(1,np.max(ind_matched0))) - set(ind_matched0)

    leftover_labels = leftover_labels.union(set(np.max(ind_matched0)+1+np.arange(len(ind_unmatched)-len(leftover_labels))))

    assert len(leftover_labels)>= len(ind_unmatched)

    
    print(f"matched: {len(ind_matched)} unmatched: {len(ind_unmatched)}")
    u = np.zeros_like(y)
    for ind0,ind in zip(ind_matched0, ind_matched):
        u[y==ind] = ind0

    for ind,ind2 in zip(ind_unmatched, leftover_labels):
        u[y==ind] = ind2
    
    return u



def render_label_pred(y_true, y_pred,
                      img = None, cmap_img = "gray", normalize_img = True, 
                      tp_alpha = .6, fp_alpha = .6, fn_alpha = .6,
                      matching_kwargs = dict(thresh=0.5)):
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
    """
    
    from matplotlib import cm

    assert y_true.shape == y_pred.shape
    
    matching_kwargs["report_matches"] = True
    res = matching(y_true, y_pred, **matching_kwargs)

    all_true = set(np.unique(y_true))-{0}
    all_pred = set(np.unique(y_pred))-{0}

    pairs = np.array(res.matched_pairs)
    scores = np.array(res.matched_scores)
    ind_tp_pairs = np.where(scores>=matching_kwargs["thresh"])[0]
    tp_true, tp_pred = tuple(zip(*pairs[ind_tp_pairs]))
    tp = tp_pred
    fn = all_true.difference(tp_true)
    fp = all_pred.difference(tp_pred)

    assert res.tp == len(tp)
    assert res.fp == len(fp)
    assert res.fn == len(fn)

    mask_tp = np.isin(y_pred, tuple(tp))
    mask_fn = np.isin(y_true, tuple(fn))
    mask_fp = np.isin(y_pred, tuple(fp))

    def gen_maps(n, h0,l0,s0):
        h,l,s = random_hls(n,h0,l0,s0)
        return cmap_from_hls(h,l,s), cmap_from_hls(h,1.4*l,s)

    n0 = np.max(y_pred)+1

    # green
    cmap_tp, cmap_border_tp = gen_maps(n0, h0 = (.25,.35) , l0 = (.4,.6), s0 = (.5,.7))
    # red
    cmap_fp, cmap_border_fp = gen_maps(n0, h0 = (0,.1) , l0 = (.4,.6), s0 = (.5,.7))
    # blue
    cmap_fn, cmap_border_fn = gen_maps(n0, h0 = (.6,.7) , l0 = (.4,.6), s0 = (.5,.7))

    im_tp = cmap_tp(y_pred)
    im_fp = cmap_fp(y_pred)
    im_fn = cmap_fn(y_true)

    # render image if given
    if img is None:
        im_img = np.zeros(y_true.shape+(4,),np.float32)
        im_img[...,-1] = 1
    else:
        assert y_true.shape[:2] == img.shape[:2]
        img = normalize(img) if normalize_img else img
        cmap_img = cm.get_cmap(cmap_img) if isinstance(cmap_img, str) else cmap_img
        if img.ndim==2:
            im_img = cmap_img(img)
        elif img.ndim==3:
            im_img = img[...,:4]
            if img.shape[-1]<4:
                im_img = np.concatenate([img, np.ones(img.shape[:2]+(4-img.shape[-1],))], axis = -1)
        else:
            raise ValueError("img should be 2 or 3 dimensional")
        
        
    # blend
    im = im_img.copy()
    
    im[mask_tp] = tp_alpha*im_tp[mask_tp]+(1-tp_alpha)*im_img[mask_tp]
    im[mask_fp] = fp_alpha*im_fp[mask_fp]+(1-fp_alpha)*im_img[mask_fp]
    im[mask_fn] = fn_alpha*im_fn[mask_fn]+(1-fn_alpha)*im_img[mask_fn]
    return im
