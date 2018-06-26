from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import numpy as np
import warnings
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage.measurements import find_objects
from skimage.draw import polygon
from csbdeep.utils import _raise


_ocl_kernel = r"""
#ifndef M_PI
#define M_PI 3.141592653589793
#endif

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline float2 pol2cart(const float rho, const float phi) {
    const float x = rho * cos(phi);
    const float y = rho * sin(phi);
    return (float2)(x,y);
}

__kernel void star_dist(__global float* dst, read_only image2d_t src) {

    const int i = get_global_id(0), j = get_global_id(1);
    const int Nx = get_global_size(0), Ny = get_global_size(1);

    const float2 origin = (float2)(i,j);
    const int value = read_imageui(src,sampler,origin).x;

    if (value == 0) {
        // background pixel -> nothing to do, write all zeros
        for (int k = 0; k < N_RAYS; k++) {
            dst[k + i*N_RAYS + j*N_RAYS*Nx] = 0;
        }
    } else {
        float st_rays = (2*M_PI) / N_RAYS; // step size for ray angles
        // for all rays
        for (int k = 0; k < N_RAYS; k++) {
            const float phi = k*st_rays; // current ray angle phi
            const float2 dir = pol2cart(1,phi); // small vector in direction of ray
            float2 offset = 0; // offset vector to be added to origin
            // find radius that leaves current object
            while (1) {
                offset += dir;
                const int offset_value = read_imageui(src,sampler,round(origin+offset)).x;
                if (offset_value != value) {
                    const float dist = sqrt(offset.x*offset.x + offset.y*offset.y);
                    dst[k + i*N_RAYS + j*N_RAYS*Nx] = dist;
                    break;
                }
            }
        }
    }

}
"""


def _ocl_star_dist(a, n_rays=32):
    from gputools import OCLProgram, OCLArray, OCLImage
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
    n_rays = int(n_rays)
    src = OCLImage.from_array(a.astype(np.uint16,copy=False))
    dst = OCLArray.empty(a.shape+(n_rays,), dtype=np.float32)
    program = OCLProgram(src_str=_ocl_kernel, build_options=['-D', 'N_RAYS=%d' % n_rays])
    program.run_kernel('star_dist', src.shape, None, dst.data, src)
    return dst.get()


def _cpp_star_dist(a, n_rays=32):
    from .lib.stardist import c_star_dist
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
    return c_star_dist(a.astype(np.uint16,copy=False), int(n_rays))


def _py_star_dist(a, n_rays=32):
    (np.isscalar(n_rays) and 0 < int(n_rays)) or _raise(ValueError())
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
                            dist = np.sqrt(x*x + y*y)
                            dst[i,j,k] = dist
                            break
    return dst


def star_dist(a, n_rays=32, opencl=False):
    """'a' assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""
    if not _is_power_of_2(n_rays):
        warnings.warn("not tested with 'n_rays' not being a power of 2.")
    if opencl:
        try:
            return _ocl_star_dist(a,n_rays)
        except:
            pass
    return _cpp_star_dist(a,n_rays)


def _is_power_of_2(i):
    assert i > 0
    e = np.log2(i)
    return e == int(e)


def ray_angles(n_rays=32):
    return np.linspace(0,2*np.pi,n_rays,endpoint=False)


def dist_to_coord(rhos):
    """convert from polar to cartesian coordinates for a single image (3-D array) or multiple images (4-D array)"""

    is_single_image = rhos.ndim == 3
    if is_single_image:
        rhos = np.expand_dims(rhos,0)
    assert rhos.ndim == 4

    n_images,h,w,n_rays = rhos.shape
    coord = np.empty((n_images,h,w,2,n_rays),dtype=rhos.dtype)

    start = np.meshgrid(np.arange(h),np.arange(w), indexing='ij')
    for i in range(2):
        start[i] = start[i].reshape(1,h,w,1)
        # start[i] = np.tile(start[i],(n_images,1,1,n_rays))
        start[i] = np.broadcast_to(start[i],(n_images,h,w,n_rays))
        coord[...,i,:] = start[i]

    phis = ray_angles(n_rays).reshape(1,1,1,n_rays)

    coord[...,0,:] += rhos * np.sin(phis) # row coordinate
    coord[...,1,:] += rhos * np.cos(phis) # col coordinate

    return coord[0] if is_single_image else coord


def _edt_prob(lbl_img):
    prob = np.zeros(lbl_img.shape,np.float32)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        edt = distance_transform_edt(mask)[mask]
        prob[mask] = edt/np.max(edt)
    return prob


def edt_prob(lbl_img):
    """Perform EDT on each labeled object and normalize."""
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
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
        edt = distance_transform_edt(grown_mask)[shrink_slice][mask]
        prob[sl][mask] = edt/np.max(edt)
    return prob


def polygons_to_label(coord, prob, points, thr=-np.inf):
    sh = coord.shape[:2]
    lbl = np.zeros(sh,np.uint16)
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
