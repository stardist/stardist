import numpy as np
from numbers import Number

def max_sparsify(x, size=2, cval=0):
    # retains only the maximum value in a neighborhood of size, setting the rest with fill_value
    # (inplace)
    from ..lib.starfinity import c_filter_max_inplace3D, c_filter_max_inplace2D 
    
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not x.ndim in(2,3):
        raise ValueError("x should be 2 or 3 dimensional") 
    
    if isinstance(size, Number):
        size = (size,)*x.ndim
    
    if not len(size) == x.ndim:
        raise ValueError(f"size should be a number or a tuple of length {x.ndim}")
    
    if x.ndim==2:
        y = c_filter_max_inplace2D(x, np.int32(size[0]), np.int32(size[1]), np.float32(cval))
    else:
        y = c_filter_max_inplace3D(x, np.int32(size[0]), np.int32(size[1]), np.int32(size[2]), np.float32(cval))
        
    return y
    
    

def dist_to_affinity2D(dist, weights = None, decay = 0, normed = False, grid = (1,1), verbose = True):
    """
    creates affinities from dist array

    dist.shape    = (y, x, d)
    weights.shape = (y, x) 
    
    affinities at distance d are weighted ~ exp(-decay*d) (set 0 -> no decay)

    """

    
    dist = np.asanyarray(dist)

    if dist.ndim != 3:
        raise ValueError("dist should be 3 dimensional but has shape %s" % str(dist.shape))

    if weights is None:
        weights = np.ones(dist.shape[:-1], np.float32)
    else:
        weights = np.asanyarray(weights)

    if weights.ndim != 2:
        raise ValueError("weights should be 2 dimensional but has shape %s" % str(weights.shape))

    if dist.shape[:-1] != weights.shape: 
        raise ValueError("first 2 axis of dist and weights should agree! (%s != %s)" % (str(dist.shape[:-1].shape),str(weights.shape)))
                
    def _prep(x,dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    from ..lib.starfinity import c_dist_to_affinity2D

    return c_dist_to_affinity2D(_prep(dist,np.float32),
                                _prep(weights,np.float32),
                                np.float32(decay),
                                np.int32(normed),
                                np.int32(grid[0]),
                                np.int32(grid[1]),
                                np.int32(verbose))


def dist_to_affinity3D(dist, rays, weights = None, decay = 0, normed = False, clip_dist = np.inf, grid = (1,1,1), verbose = True):
    """
    creates affinities from dist array

    dist.shape    = (z, y, x, d)
    weights.shape = (z, y, x) 

    affinities along a ray are decayed by ~ exp(-decay*d) (set 0 -> no decay)

    """

    
    dist = np.asanyarray(dist)

    if dist.ndim != 4:
        raise ValueError("dist should be 4 dimensional but has shape %s" % str(dist.shape))

    if weights is None:
        weights = np.ones(dist.shape[:-1], np.float32)
    else:
        weights = np.asanyarray(weights)

    if weights.ndim != 3:
        raise ValueError("weights should be 3 dimensional but has shape %s" % str(weights.shape))
        
    if dist.shape[:-1] != weights.shape: 
        raise ValueError("first 3 axis of dist and weights should agree! (%s != %s)" % (str(dist.shape[:-1].shape),str(weights.shape)))

    
    def _prep(x,dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    from ..lib.starfinity import c_dist_to_affinity3D

    return c_dist_to_affinity3D(_prep(dist,np.float32),
                                _prep(rays.vertices,np.float32),
                                _prep(weights,np.float32),
                                np.float32(decay),
                                np.int32(normed),
                                np.float32(clip_dist),
                                np.int32(grid[0]),
                                np.int32(grid[1]),
                                np.int32(grid[2]),
                                np.int32(verbose))


