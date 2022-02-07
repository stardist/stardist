"""
Ray factory

classes that provide vertex and triangle information for rays on spheres

Example:

    rays = Rays_Tetra(n_level = 4)

    print(rays.vertices)
    print(rays.faces)

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from scipy.spatial import ConvexHull
import copy
import warnings

class Rays_Base(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._vertices, self._faces = self.setup_vertices_faces()
        self._vertices = np.asarray(self._vertices, np.float32)
        self._faces = np.asarray(self._faces, int)
        self._faces = np.asanyarray(self._faces)

    def setup_vertices_faces(self):
        """has to return

         verts , faces

         verts = ( (z_1,y_1,x_1), ... )
         faces ( (0,1,2), (2,3,4), ... )

         """
        raise NotImplementedError()

    @property
    def vertices(self):
        """read-only property"""
        return self._vertices.copy()

    @property
    def faces(self):
        """read-only property"""
        return self._faces.copy()

    def __getitem__(self, i):
        return self.vertices[i]

    def __len__(self):
        return len(self._vertices)

    def __repr__(self):
        def _conv(x):
            if isinstance(x,(tuple, list, np.ndarray)):
                return "_".join(_conv(_x) for _x in x)
            if isinstance(x,float):
                return "%.2f"%x
            return str(x)
        return "%s_%s" % (self.__class__.__name__, "_".join("%s_%s" % (k, _conv(v)) for k, v in sorted(self.kwargs.items())))
    
    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "kwargs": self.kwargs
        }

    def dist_loss_weights(self, anisotropy = (1,1,1)):
        """returns the anisotropy corrected weights for each ray"""
        anisotropy = np.array(anisotropy)
        assert anisotropy.shape == (3,)
        return np.linalg.norm(self.vertices*anisotropy, axis = -1)

    def volume(self, dist=None):
        """volume of the starconvex polyhedron spanned by dist (if None, uses dist=1)
        dist can be a nD array, but the last dimension has to be of length n_rays
        """
        if dist is None: dist = np.ones_like(self.vertices)

        dist = np.asarray(dist)
        
        if not dist.shape[-1]==len(self.vertices):
            raise ValueError("last dimension of dist should have length len(rays.vertices)")
        # all the shuffling below is to allow dist to be an arbitrary sized array (with last dim n_rays)
        # self.vertices -> (n_rays,3)
        # dist -> (m,n,..., n_rays)
        
        # dist  -> (m,n,..., n_rays, 3)
        dist = np.repeat(np.expand_dims(dist,-1), 3, axis = -1)
        # verts  -> (m,n,..., n_rays, 3)
        verts = np.broadcast_to(self.vertices, dist.shape)

        # dist, verts  -> (n_rays, m,n, ..., 3)        
        dist = np.moveaxis(dist,-2,0)
        verts = np.moveaxis(verts,-2,0)

        # vs -> (n_faces, 3, m, n, ..., 3)
        vs = (dist*verts)[self.faces]
        # vs -> (n_faces, m, n, ..., 3, 3)
        vs = np.moveaxis(vs, 1,-2)
        # vs -> (n_faces * m * n, 3, 3)        
        vs = vs.reshape((len(self.faces)*int(np.prod(dist.shape[1:-1])),3,3))
        d = np.linalg.det(list(vs)).reshape((len(self.faces),)+dist.shape[1:-1])
        
        return -1./6*np.sum(d, axis = 0)
    
    def surface(self, dist=None):
        """surface area of the starconvex polyhedron spanned by dist (if None, uses dist=1)"""
        dist = np.asarray(dist)
        
        if not dist.shape[-1]==len(self.vertices):
            raise ValueError("last dimension of dist should have length len(rays.vertices)")

        # self.vertices -> (n_rays,3)
        # dist -> (m,n,..., n_rays)
        
        # all the shuffling below is to allow dist to be an arbitrary sized array (with last dim n_rays)
        
        # dist  -> (m,n,..., n_rays, 3)
        dist = np.repeat(np.expand_dims(dist,-1), 3, axis = -1)
        # verts  -> (m,n,..., n_rays, 3)
        verts = np.broadcast_to(self.vertices, dist.shape)

        # dist, verts  -> (n_rays, m,n, ..., 3)        
        dist = np.moveaxis(dist,-2,0)
        verts = np.moveaxis(verts,-2,0)

        # vs -> (n_faces, 3, m, n, ..., 3)
        vs = (dist*verts)[self.faces]
        # vs -> (n_faces, m, n, ..., 3, 3)
        vs = np.moveaxis(vs, 1,-2)
        # vs -> (n_faces * m * n, 3, 3)        
        vs = vs.reshape((len(self.faces)*int(np.prod(dist.shape[1:-1])),3,3))
       
        pa = vs[...,1,:]-vs[...,0,:]
        pb = vs[...,2,:]-vs[...,0,:]

        d = .5*np.linalg.norm(np.cross(list(pa), list(pb)), axis = -1)
        d = d.reshape((len(self.faces),)+dist.shape[1:-1])
        return np.sum(d, axis = 0)

    
    def copy(self, scale=(1,1,1)):
        """ returns a copy whose vertices are scaled by given factor"""
        scale = np.asarray(scale)
        assert scale.shape == (3,)
        res = copy.deepcopy(self)
        res._vertices *= scale[np.newaxis]
        return res 



    
def rays_from_json(d):
    return eval(d["name"])(**d["kwargs"])


################################################################

class Rays_Explicit(Rays_Base):
    def __init__(self, vertices0, faces0):
        self.vertices0, self.faces0 = vertices0, faces0
        super().__init__(vertices0=list(vertices0), faces0=list(faces0))
        
    def setup_vertices_faces(self):
        return self.vertices0, self.faces0
    

class Rays_Cartesian(Rays_Base):
    def __init__(self, n_rays_x=11, n_rays_z=5):
        super().__init__(n_rays_x=n_rays_x, n_rays_z=n_rays_z)

    def setup_vertices_faces(self):
        """has to return list of ( (z_1,y_1,x_1), ... )  _"""
        n_rays_x, n_rays_z = self.kwargs["n_rays_x"], self.kwargs["n_rays_z"]
        dphi = np.float32(2. * np.pi / n_rays_x)
        dtheta = np.float32(np.pi / n_rays_z)

        verts = []
        for mz in range(n_rays_z):
            for mx in range(n_rays_x):
                phi = mx * dphi
                theta = mz * dtheta
                if mz == 0:
                    theta = 1e-12
                if mz == n_rays_z - 1:
                    theta = np.pi - 1e-12
                dx = np.cos(phi) * np.sin(theta)
                dy = np.sin(phi) * np.sin(theta)
                dz = np.cos(theta)
                if mz == 0 or mz == n_rays_z - 1:
                    dx += 1e-12
                    dy += 1e-12
                verts.append([dz, dy, dx])

        verts = np.array(verts)

        def _ind(mz, mx):
            return mz * n_rays_x + mx

        faces = []

        for mz in range(n_rays_z - 1):
            for mx in range(n_rays_x):
                faces.append([_ind(mz, mx), _ind(mz + 1, (mx + 1) % n_rays_x), _ind(mz, (mx + 1) % n_rays_x)])
                faces.append([_ind(mz, mx), _ind(mz + 1, mx), _ind(mz + 1, (mx + 1) % n_rays_x)])

        faces = np.array(faces)

        return verts, faces


class Rays_SubDivide(Rays_Base):
    """
    Subdivision polyehdra

    n_level = 1 -> base polyhedra
    n_level = 2 -> 1x subdivision
    n_level = 3 -> 2x subdivision
                ...
    """

    def __init__(self, n_level=4):
        super().__init__(n_level=n_level)

    def base_polyhedron(self):
        raise NotImplementedError()

    def setup_vertices_faces(self):
        n_level = self.kwargs["n_level"]
        verts0, faces0 = self.base_polyhedron()
        return self._recursive_split(verts0, faces0, n_level)

    def _recursive_split(self, verts, faces, n_level):
        if n_level <= 1:
            return verts, faces
        else:
            verts, faces = Rays_SubDivide.split(verts, faces)
            return self._recursive_split(verts, faces, n_level - 1)

    @classmethod
    def split(self, verts0, faces0):
        """split a level"""

        split_edges = dict()
        verts = list(verts0[:])
        faces = []

        def _add(a, b):
            """ returns index of middle point and adds vertex if not already added"""
            edge = tuple(sorted((a, b)))
            if not edge in split_edges:
                v = .5 * (verts[a] + verts[b])
                v *= 1. / np.linalg.norm(v)
                verts.append(v)
                split_edges[edge] = len(verts) - 1
            return split_edges[edge]

        for v1, v2, v3 in faces0:
            ind1 = _add(v1, v2)
            ind2 = _add(v2, v3)
            ind3 = _add(v3, v1)
            faces.append([v1, ind1, ind3])
            faces.append([v2, ind2, ind1])
            faces.append([v3, ind3, ind2])
            faces.append([ind1, ind2, ind3])

        return verts, faces


class Rays_Tetra(Rays_SubDivide):
    """
    Subdivision of a tetrahedron

    n_level = 1 -> normal tetrahedron (4 vertices)
    n_level = 2 -> 1x subdivision (10 vertices)
    n_level = 3 -> 2x subdivision (34 vertices)
                ...
    """

    def base_polyhedron(self):
        verts = np.array([
            [np.sqrt(8. / 9), 0., -1. / 3],
            [-np.sqrt(2. / 9), np.sqrt(2. / 3), -1. / 3],
            [-np.sqrt(2. / 9), -np.sqrt(2. / 3), -1. / 3],
            [0., 0., 1.]
        ])
        faces = [[0, 1, 2],
                 [0, 3, 1],
                 [0, 2, 3],
                 [1, 3, 2]]

        return verts, faces


class Rays_Octo(Rays_SubDivide):
    """
    Subdivision of a tetrahedron

    n_level = 1 -> normal Octahedron (6 vertices)
    n_level = 2 -> 1x subdivision (18 vertices)
    n_level = 3 -> 2x subdivision (66 vertices)

    """

    def base_polyhedron(self):
        verts = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, -1],
            [0, -1, 0],
            [1, 0, 0],
            [-1, 0, 0]])

        faces = [[0, 1, 4],
                 [0, 5, 1],
                 [1, 2, 4],
                 [1, 5, 2],
                 [2, 3, 4],
                 [2, 5, 3],
                 [3, 0, 4],
                 [3, 5, 0],
                 ]

        return verts, faces


def reorder_faces(verts, faces):
    """reorder faces such that their orientation points outward"""
    def _single(face):
        return face[::-1] if np.linalg.det(verts[face])>0 else face
    return tuple(map(_single, faces))


class Rays_GoldenSpiral(Rays_Base):
    def __init__(self, n=70, anisotropy = None):
        if n<4:
            raise ValueError("At least 4 points have to be given!")
        super().__init__(n=n, anisotropy = anisotropy if anisotropy is None else tuple(anisotropy))

    def setup_vertices_faces(self):
        n = self.kwargs["n"]
        anisotropy = self.kwargs["anisotropy"]
        if anisotropy is None:
            anisotropy = np.ones(3)
        else:
            anisotropy = np.array(anisotropy)

        # the smaller golden angle = 2pi * 0.3819...
        g = (3. - np.sqrt(5.)) * np.pi
        phi = g * np.arange(n)
        # z = np.linspace(-1, 1, n + 2)[1:-1]
        # rho = np.sqrt(1. - z ** 2)
        # verts = np.stack([rho*np.cos(phi), rho*np.sin(phi),z]).T
        #
        z = np.linspace(-1, 1, n)
        rho = np.sqrt(1. - z ** 2)
        verts = np.stack([z, rho * np.sin(phi), rho * np.cos(phi)]).T

        # warnings.warn("ray definition has changed! Old results are invalid!")

        # correct for anisotropy
        verts = verts/anisotropy
        #verts /= np.linalg.norm(verts, axis=-1, keepdims=True)

        hull = ConvexHull(verts)
        faces = reorder_faces(verts,hull.simplices)

        verts /= np.linalg.norm(verts, axis=-1, keepdims=True)

        return verts, faces
