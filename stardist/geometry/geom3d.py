from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import os

from skimage.measure import regionprops
from csbdeep.utils import _raise
from tqdm import tqdm

from ..utils import path_absolute, _normalize_grid
from ..matching import _check_label_array
# from ..lib.stardist3d import c_star_dist3d, c_polyhedron_to_label, c_dist_to_volume, c_dist_to_centroid
from ..lib.stardist3d import c_star_dist3d, c_polyhedron_to_label



def _cpp_star_dist3D(lbl, rays, grid=(1,1,1)):
    dz, dy, dx = rays.vertices.T
    grid = _normalize_grid(grid,3)

    return c_star_dist3d(lbl.astype(np.uint16, copy=False),
                         dz.astype(np.float32, copy=False),
                         dy.astype(np.float32, copy=False),
                         dx.astype(np.float32, copy=False),
                         int(len(rays)), *tuple(int(a) for a in grid))


def _py_star_dist3D(img, rays, grid=(1,1,1)):
    grid = _normalize_grid(grid,3)
    img = img.astype(np.uint16, copy=False)
    dst_shape = tuple(s // a for a, s in zip(grid, img.shape)) + (len(rays),)
    dst = np.empty(dst_shape, np.float32)

    dzs, dys, dxs = rays.vertices.T

    for i in range(dst_shape[0]):
        for j in range(dst_shape[1]):
            for k in range(dst_shape[2]):
                value = img[i * grid[0], j * grid[1], k * grid[2]]
                if value == 0:
                    dst[i, j, k] = 0
                else:

                    for n, (dz, dy, dx) in enumerate(zip(dzs, dys, dxs)):
                        x, y, z = np.float32(0), np.float32(0), np.float32(0)
                        while True:
                            x += dx
                            y += dy
                            z += dz
                            ii = int(round(i * grid[0] + z))
                            jj = int(round(j * grid[1] + y))
                            kk = int(round(k * grid[2] + x))
                            if (ii < 0 or ii >= img.shape[0] or
                                        jj < 0 or jj >= img.shape[1] or
                                        kk < 0 or kk >= img.shape[2] or
                                        value != img[ii, jj, kk]):
                                dist = np.sqrt(x * x + y * y + z * z)
                                dst[i, j, k, n] = dist
                                break

    return dst


def _ocl_star_dist3D(lbl, rays, grid=(1,1,1)):
    from gputools import OCLProgram, OCLArray, OCLImage

    grid = _normalize_grid(grid,3)

    # if not all(g==1 for g in grid):
    #     raise NotImplementedError("grid not yet implemented for OpenCL version of star_dist3D()...")

    # slicing with grid is done with tuple(slice(0, None, g) for g in grid)
    res_shape = tuple((s-1)//g+1 for s, g in zip(lbl.shape, grid))

    lbl_g = OCLImage.from_array(lbl.astype(np.uint16, copy=False))
    dist_g = OCLArray.empty(res_shape + (len(rays),), dtype=np.float32)
    rays_g = OCLArray.from_array(rays.vertices.astype(np.float32, copy=False))

    program = OCLProgram(path_absolute("kernels/stardist3d.cl"), build_options=['-D', 'N_RAYS=%d' % len(rays)])
    program.run_kernel('stardist3d', res_shape[::-1], None,
                       lbl_g, rays_g.data, dist_g.data,
                       np.int32(grid[0]),np.int32(grid[1]),np.int32(grid[2]))

    return dist_g.get()


def star_dist3D(lbl, rays, grid=(1,1,1), mode='cpp'):
    """lbl assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""

    grid = _normalize_grid(grid,3)
    if mode == 'python':
        return _py_star_dist3D(lbl, rays, grid=grid)
    elif mode == 'cpp':
        return _cpp_star_dist3D(lbl, rays, grid=grid)
    elif mode == 'opencl':
        return _ocl_star_dist3D(lbl, rays, grid=grid)
    else:
        _raise(ValueError("Unknown mode %s" % mode))


def polyhedron_to_label(dist, points, rays, shape, prob=None, thr=-np.inf, labels=None, mode="full", verbose=True, overlap_label=None):
    """
    creates labeled image from stardist representations

    :param dist: array of shape (n_points,n_rays)
        the list of distances for each point and ray
    :param points: array of shape (n_points, 3)
        the list of center points
    :param rays: Rays object
        Ray object (e.g. `stardist.Rays_GoldenSpiral`) defining
        vertices and faces
    :param shape: (nz,ny,nx)
        output shape of the image
    :param prob: array of length/shape (n_points,) or None
        probability per polyhedron
    :param thr: scalar
        probability threshold (only polyhedra with prob>thr are labeled)
    :param labels: array of length/shape (n_points,) or None
        labels to use
    :param mode: str
        labeling mode, can be "full", "kernel", "hull", "bbox" or  "debug"
    :param verbose: bool
        enable to print some debug messages
    :param overlap_label: scalar or None
        if given, will label each pixel that belongs ot more than one polyhedron with that label
    :return: array of given shape
        labeled image
    """
    if len(points) == 0:
        if verbose:
            print("warning: empty list of points (returning background-only image)")
        return np.zeros(shape, np.uint16)

    dist = np.asanyarray(dist)
    points = np.asanyarray(points)

    if dist.ndim == 1:
        dist = dist.reshape(1, -1)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if labels is None:
        labels = np.arange(1, len(points) + 1)

    if np.amin(dist) <= 0:
        raise ValueError("distance array should be positive!")

    prob = np.ones(len(points)) if prob is None else np.asanyarray(prob)

    if dist.ndim != 2:
        raise ValueError("dist should be 2 dimensional but has shape %s" % str(dist.shape))

    if dist.shape[1] != len(rays):
        raise ValueError("inconsistent number of rays!")

    if len(prob) != len(points):
        raise ValueError("len(prob) != len(points)")

    if len(labels) != len(points):
        raise ValueError("len(labels) != len(points)")

    modes = {"full": 0, "kernel": 1, "hull": 2, "bbox": 3, "debug": 4}

    if not mode in modes:
        raise KeyError("Unknown render mode '%s' , allowed:  %s" % (mode, tuple(modes.keys())))

    lbl = np.zeros(shape, np.uint16)

    # filter points
    ind = np.where(prob >= thr)[0]
    if len(ind) == 0:
        if verbose:
            print("warning: no points found with probability>= {thr:.4f} (returning background-only image)".format(thr=thr))
        return lbl

    prob = prob[ind]
    points = points[ind]
    dist = dist[ind]
    labels = labels[ind]

    # sort points with decreasing probability
    ind = np.argsort(prob)[::-1]
    points = points[ind]
    dist = dist[ind]
    labels = labels[ind]

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    return c_polyhedron_to_label(_prep(dist, np.float32),
                                 _prep(points, np.float32),
                                 _prep(rays.vertices, np.float32),
                                 _prep(rays.faces, np.int32),
                                 _prep(labels, np.int32),
                                 np.int32(modes[mode]),
                                 np.int32(verbose),
                                 np.int32(overlap_label is not None),
                                 np.int32(0 if overlap_label is None else overlap_label),
                                 shape
                                 )


def relabel_image_stardist3D(lbl, rays, verbose=False, **kwargs):
    """relabel each label region in `lbl` with its star representation"""
    _check_label_array(lbl, "lbl")
    if not lbl.ndim==3:
        raise ValueError("lbl image should be 3 dimensional")

    dist_all = star_dist3D(lbl, rays, **kwargs)

    regs = regionprops(lbl)

    points = np.array(tuple(np.array(r.centroid).astype(int) for r in regs))
    labs = np.array(tuple(r.label for r in regs))
    dist = np.array(tuple(dist_all[p[0], p[1], p[2]] for p in points))
    dist = np.maximum(dist, 1e-3)

    lbl_new = polyhedron_to_label(dist, points, rays, shape=lbl.shape, labels=labs, verbose=verbose)
    return lbl_new


def dist_to_volume(dist, rays):
    """ returns areas of polyhedra
        dist.shape = (nz,ny,nx,nrays)
    """
    dist = np.asanyarray(dist)
    dist.ndim == 4 or _raise(ValueError("dist.ndim = {dist.ndim} but should be 4".format(dist = dist)))
    dist.shape[-1]== len(rays) or _raise(ValueError("dist.shape[-1] = {d} but should be {l}".format(d = dist.shape[-1], l = len(rays))))

    dist = np.ascontiguousarray(dist.astype(np.float32, copy=False))

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    return c_dist_to_volume(_prep(dist, np.float32),
                          _prep(rays.vertices, np.float32),
                          _prep(rays.faces, np.int32))


def dist_to_centroid(dist, rays, mode='absolute'):
    """ returns centroids of polyhedra

        dist.shape = (nz,ny,nx,nrays)
        mode = 'absolute' or 'relative'

    """
    dist.ndim == 4 or _raise(ValueError("dist.ndim = {dist.ndim} but should be 4".format(dist = dist)))
    dist.shape[-1]== len(rays) or _raise(ValueError("dist.shape[-1] = {d} but should be {l}".format(d = dist.shape[-1], l = len(rays))))
    dist = np.ascontiguousarray(dist.astype(np.float32, copy=False))

    mode in ('absolute', 'relative') or _raise(ValueError("mode should be either 'absolute' or 'relative'"))

    def _prep(x, dtype):
        return np.ascontiguousarray(x.astype(dtype, copy=False))

    return c_dist_to_centroid(_prep(dist, np.float32),
                          _prep(rays.vertices, np.float32),
                          _prep(rays.faces, np.int32),
                              np.int32(mode=='absolute'))



def dist_to_coord3D(dist, points, rays_vertices):
    """ converts dist/points/rays_vertices to list of coords """

    dist = np.asarray(dist)
    points = np.asarray(points)
    rays_vertices = np.asarray(rays_vertices)

    if not all((len(dist)==len(points), dist.ndim==2, points.ndim==2,
               points.shape[-1]==3, rays_vertices.shape[-1]==3, dist.shape[-1]==len(rays_vertices))):
        raise ValueError(f"Wrong shapes! dist -> (m,n) points -> (m,3) rays_vertices -> (m,)")

    # return points[:,np.newaxis]+dist[...,np.newaxis]*rays_vertices

    return points[:,np.newaxis]+dist[...,np.newaxis]*rays_vertices


def export_to_obj_file3D(polys, fname=None, scale=1, single_mesh=True, uv_map=False, name="poly"):
    """ exports 3D mesh result to obj file format """

    try:
        dist = polys["dist"]
        points = polys["points"]
        rays_vertices = polys["rays_vertices"]
        rays_faces = polys["rays_faces"]
    except KeyError as e:
        print(e)
        raise ValueError("polys should be a dict with keys 'dist', 'points', 'rays_vertices', 'rays_faces'  (such as generated by StarDist3D.predict_instances) ")

    coord = dist_to_coord3D(dist, points, rays_vertices)

    if not all((coord.ndim==3, coord.shape[-1]==3, rays_faces.shape[-1]==3)):
        raise ValueError(f"Wrong shapes! coord -> (m,n,3) rays_faces -> (k,3)")

    if np.isscalar(scale):
        scale = (scale,)*3

    scale = np.asarray(scale)
    assert len(scale)==3

    coord *= scale

    obj_str = ""
    vert_count = 0

    decimals = int(max(1,1-np.log10(np.min(scale))))


    scaled_verts = scale*rays_vertices
    scaled_verts /= np.linalg.norm(scaled_verts,axis = 1, keepdims=True)


    vertex_line = f"v {{x:.{decimals}f}} {{y:.{decimals}f}} {{z:.{decimals}f}}\n"

    rays_faces = rays_faces.copy()+1

    for i, xs in enumerate(tqdm(coord)):
        # reorder to xyz
        xs = xs[:,[2,1,0]]

        # print(xs)

        # new object
        if i==0 or not single_mesh:
            obj_str += f"o {name}_{i:d}\n"

        # vertex coords
        for x,y,z in xs:
            obj_str += vertex_line.format(x=x,y=y,z=z)

        if uv_map:
            # UV coords
            for vz,vy,vx in scaled_verts:
                u = 1-(.5 + .5*np.arctan2(vz,vx)/np.pi)
                v = 1-(.5 - np.arcsin(vy)/np.pi)
                obj_str +=  f"vt {u:.4f} {v:.4f}\n"

        # face indices
        for face in rays_faces:
            obj_str += f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n"

        rays_faces += len(xs)

    if fname is not None:
        with open(fname,"w") as f:
            f.write(obj_str)

    return obj_str


