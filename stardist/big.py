import numpy as np
import warnings
import math
from tqdm import tqdm
from skimage.measure import regionprops
from skimage.draw import polygon
from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict
from itertools import product

from .matching import relabel_sequential
from .geometry import polyhedron_to_label


class Tile:

    def __init__(self, size, min_overlap, context, pred):
        self.size = int(size)
        self.min_overlap = int(min_overlap)
        self.context = int(context)
        self.pred = pred
        self.succ = None
        assert 0 <= self.min_overlap + 2*self.context < self.size
        self.stride = self.size - (self.min_overlap + 2*self.context)
        self._start = 0
        self._frozen = False

    @property
    def start(self):
        return self._start if (self.frozen or self.at_begin) else self.pred.succ_start

    @property
    def end(self):
        return self.start + self.size

    @property
    def succ_start(self):
        return self.start + self.stride

    def add_succ(self):
        assert self.succ is None and not self.frozen
        self.succ = Tile(self.size, self.min_overlap, self.context, self)
        return self.succ

    def decrease_stride(self, amount):
        amount = int(amount)
        assert 0 <= amount < self.stride and not self.frozen
        self.stride -= amount

    def freeze(self):
        assert not self.frozen and (self.at_begin or self.pred.frozen)
        self._start = self.start
        self._frozen = True
        if not self.at_end:
            self.succ.freeze()

    @property
    def slice_read(self):
        return slice(self.start, self.end)

    @property
    def slice_crop_context(self):
        return slice(self.context_start, self.size - self.context_end)

    @property
    def slice_write(self):
        return slice(self.start + self.context_start, self.end - self.context_end)

    def is_responsible(self, bbox):
        # bbox is the 1D bounding box / interval
        # coordinates are relative to size without context
        bmin, bmax = bbox

        r_start = 0 if self.at_begin else (self.pred.overlap - self.pred.context_end - self.context_start)
        r_end = self.size - self.context_start - self.context_end
        assert 0 <= bmin < bmax <= r_end

        # assert not (bmin == 0 and bmax >= r_start and not self.at_begin), [(r_start,r_end), bbox, self]

        if bmin == 0 and bmax >= r_start:
            if bmax == r_end:
                # object spans the entire tile, i.e. is probably larger than tile_size (minus the context)
                raise NotFullyVisible(True)
            if not self.at_begin:
                # object spans the entire overlap region, i.e. is only partially visible here and also by the predecessor tile
                raise NotFullyVisible(False)

        # object ends before responsible region start
        if bmax < r_start: return False
        # object touches the end of the responsible region (only take if at end)
        if bmax == r_end and not self.at_end: return False
        return True

    # ------------------------

    @property
    def frozen(self):
        return self._frozen

    @property
    def at_begin(self):
        return self.pred is None

    @property
    def at_end(self):
        return self.succ is None

    @property
    def overlap(self):
        return self.size - self.stride

    @property
    def context_start(self):
        return 0 if self.at_begin else self.context

    @property
    def context_end(self):
        return 0 if self.at_end else self.context

    def __repr__(self):
        shared  = f'{self.start:03}:{self.end:03}'
        shared += f', size={self.context_start}-{self.size-self.context_start-self.context_end}-{self.context_end}'
        if self.at_end:
            return f'{self.__class__.__name__}({shared})'
        else:
            return f'{self.__class__.__name__}({shared}, overlap={self.overlap}/{self.overlap-self.context_start-self.context_end})'

    @property
    def chain(self):
        tiles = [self]
        while not tiles[-1].at_end:
            tiles.append(tiles[-1].succ)
        return tiles

    def __iter__(self):
        return iter(self.chain)

    # ------------------------

    @staticmethod
    def get_tiling(size, tile_size, min_overlap, context, grid=1):

        assert 0 <= min_overlap+2*context < tile_size <= size
        assert 0 < grid <= tile_size
        tile_size = _grid_divisible(grid, tile_size, name='tile_size')
        min_overlap = _grid_divisible(grid, min_overlap, name='min_overlap')
        context = _grid_divisible(grid, context, name='context')
        assert all(v % grid == 0 for v in (tile_size, min_overlap, context))

        # allow size not to be divisible by grid
        size_orig = size
        size = _grid_divisible(grid, size, name='size', verbose=False)

        # divide all sizes by grid
        size //= grid
        tile_size //= grid
        min_overlap //= grid
        context //= grid

        # compute tiling in grid-multiples
        t = first = Tile(tile_size, min_overlap, context, None)
        while t.end < size:
            t = t.add_succ()
        last = t

        # [print(t) for t in first]

        # move tiles around to make it fit
        excess = last.end - size
        t = first
        while excess > 0:
            t.decrease_stride(1)
            excess -= 1
            t = t.succ
            if (t == last): t = first

        # make a copy of the tiling and multiply sizes by grid
        if grid > 1:
            size *= grid
            tile_size *= grid
            min_overlap *= grid
            context *= grid
            #
            _t = _first = first
            t = first = Tile(tile_size, min_overlap, context, None)
            t.stride = _t.stride*grid
            while not _t.at_end:
                _t = _t.succ
                t = t.add_succ()
                t.stride = _t.stride*grid
            last = t

            # change size of last tile
            # will be padded internally to the same size
            # as the others by model.predict_instances
            size_delta = size - size_orig
            last.size -= size_delta
            assert size_delta < grid

        # for efficiency (to not determine starts recursively from now on)
        first.freeze()

        tiles = first.chain

        # sanity checks
        assert first.start == 0 and last.end == size_orig
        assert all(t.overlap-2*context >= min_overlap for t in tiles if t != last)
        assert all (t.start % grid == 0 and t.end % grid == 0 for t in tiles if t != last)
        # print(); [print(t) for t in first]

        # only neighboring tiles should be overlapping
        if len(tiles) >= 3:
            for t in tiles[:-2]:
                assert t.slice_write.stop <= t.succ.succ.slice_write.start

        return tiles



class TileND:

    def __init__(self, id, tiles, axes):
        self.id = id
        self.tiles = tuple(tiles)
        self.axes = axes_check_and_normalize(axes, length=len(self.tiles))
        self.axis_to_tile = dict(zip(self.axes,self.tiles))

    def tiles_for_axes(self, axes=None):
        axes = self.axes if axes is None else axes_check_and_normalize(axes)
        return tuple(self.axis_to_tile[a] for a in axes)

    def slice_read(self, axes=None):
        return tuple(t.slice_read for t in self.tiles_for_axes(axes))

    def slice_crop_context(self, axes=None):
        return tuple(t.slice_crop_context for t in self.tiles_for_axes(axes))

    def slice_write(self, axes=None):
        return tuple(t.slice_write for t in self.tiles_for_axes(axes))

    def read(self, x, axes=None):
        return x[self.slice_read(axes)]

    def crop_context(self, labels, axes=None):
        return labels[self.slice_crop_context(axes)]

    def write(self, x, labels, axes=None):
        mask = labels > 0
        x[self.slice_write(axes)][mask] = labels[mask]

    def is_responsible(self, slice, axes=None):
        return all(t.is_responsible((s.start,s.stop)) for t,s in zip(self.tiles_for_axes(axes),slice))

    def __repr__(self):
        slices =  ','.join(f'{a}={t.start:03}:{t.end:03}' for t,a in zip(self.tiles,self.axes))
        return f'{self.__class__.__name__}({self.id}|{slices})'

    def __iter__(self):
        return iter(self.tiles)

    def filter_objects(self, lbl, polys, polys_sort_by='prob', axes=None):
        # todo: option to update lbl in-place
        assert np.issubdtype(lbl.dtype, np.integer)
        ndim = len(self.tiles_for_axes(axes))
        assert ndim in (2,3)
        assert lbl.ndim == ndim and lbl.shape == tuple(s.stop-s.start for s in self.slice_crop_context(axes))

        lbl_filtered = np.zeros_like(lbl)
        problem_labels = []
        for r in regionprops(lbl):
            slices = tuple(slice(r.bbox[i],r.bbox[i+lbl.ndim]) for i in range(lbl.ndim))
            try:
                if self.is_responsible(slices, axes):
                    lbl_filtered[slices][r.image] = r.label
            except NotFullyVisible as e:
                problem_labels.append(r.label)
                lbl_filtered[slices][r.image] = r.label
                # render the poly again into the label image, but this is not
                # ideal since the assumption is that the object outside that
                # region is not reliable because it's in the context
                object_larger_than_tile = e.args[0]
                if object_larger_than_tile:
                    # problem, since this object will probably be saved by another tile too
                    # raise NotImplementedError("found object larger than 'tile_size'")
                    print("found object larger than 'tile_size'")
                else:
                    # problem, but maybe fine
                    # raise NotImplementedError("found object larger than 'min_overlap'")
                    print("found object larger than 'min_overlap'")

        if polys is None:
            assert len(problem_labels) == 0
            return lbl_filtered
        else:
            # it is assumed that ids in 'lbl' map to entries in 'polys'
            coord_keys = ('points','coord') if ndim == 2 else ('points',)
            assert isinstance(polys,dict) and coord_keys is not None and all(k in polys for k in list(coord_keys)+[polys_sort_by])
            filtered_labels = np.unique(lbl_filtered)
            filtered_ind = [i-1 for i in filtered_labels if i > 0]
            polys_out = {k: (v[filtered_ind] if isinstance(v,np.ndarray) else v) for k,v in polys.items()}
            for k in coord_keys:
                polys_out[k] = self.translate_coordinates(polys_out[k], axes=axes)

        return lbl_filtered, polys_out, tuple(problem_labels)

    def translate_coordinates(self, coordinates, context=False, axes=None):
        ndim = len(self.tiles_for_axes(axes))
        assert isinstance(coordinates, np.ndarray) and coordinates.ndim >= 2 and coordinates.shape[1] == ndim
        start = [sl_read.start + bool(context)*sl_crop.start for sl_read,sl_crop in zip(self.slice_read(axes),self.slice_crop_context(axes))]
        shape = tuple(1 if d!=1 else ndim for d in range(coordinates.ndim))
        start = np.array(start).reshape(shape)
        return coordinates + start



class NotFullyVisible(Exception):
    pass



def get_tiling(shape, axes, tile_size, min_overlap, context, grid=1):
    shape = tuple(shape)
    axes = axes_check_and_normalize(axes, length=len(shape))
    n = len(shape)
    if np.isscalar(tile_size): tile_size = (tile_size,)*n
    if np.isscalar(min_overlap): min_overlap = (min_overlap,)*n
    if np.isscalar(context): context = (context,)*n
    if np.isscalar(grid): grid = (grid,)*n
    assert n == len(tile_size) == len(min_overlap) == len(context) == len(grid)

    tiling = [Tile.get_tiling(_size, _tile_size, _min_overlap, _context, _grid)
              for _size, _tile_size, _min_overlap, _context, _grid
              in  zip(shape, tile_size, min_overlap, context, grid)]

    return tuple(TileND(i,tiles,axes) for i,tiles in enumerate(product(*tiling)))



def predict_big(model, img, axes, tile_size, min_overlap, context, show_progress=True, **kwargs):
    n = img.ndim
    axes = axes_check_and_normalize(axes, length=n)

    if np.isscalar(tile_size): tile_size = (tile_size,)*n
    if np.isscalar(min_overlap): min_overlap = (min_overlap,)*n
    if np.isscalar(context): context = (context,)*n

    # TODO: do properly
    # if any(_context < (_overlap//2) for _context,_overlap in zip(context,model._axes_tile_overlap(axes))):
    #    print("context too small")

    grid = model._axes_div_by(axes)
    axes_out = model._axes_out.replace('C','')
    shape_dict = dict(zip(axes,img.shape))
    shape_out = tuple(shape_dict[a] for a in axes_out)

    if 'C' in axes:
        # TODO: tell user if values have been changed
        i = axes_dict(axes)['C']
        tile_size = list(tile_size)
        min_overlap = list(min_overlap)
        context = list(context)
        # don't tile channel axis
        tile_size[i] = img.shape[i]
        # overlap and context for channel axis not needed
        min_overlap[i] = 0
        context[i] = 0

    # print('DEBUG:')
    # print(f'img.shape = {img.shape}, axes={axes}, axes_out={axes_out}')
    # print(f'tile_size = {tile_size}, min_overlap={min_overlap}, context={context}')
    # print(flush=True)

    tiles = get_tiling(img.shape, axes, tile_size, min_overlap, context, grid)
    output = np.zeros(shape_out, dtype=np.int32)
    polys_all = {}
    labels_incomplete = []
    label_offset = 1

    if show_progress:
        tiles = tqdm(tiles)

    for tile in tiles:
        block, polys = model.predict_instances(tile.read(img, axes=axes), **kwargs)
        block = tile.crop_context(block, axes=axes_out)
        block, polys, incomplete = tile.filter_objects(block, polys, axes=axes_out)
        # TODO: replace relabel_sequential with efficient variant (had some of that in my matching lib)
        block, fwd_map, _ = relabel_sequential(block, label_offset)
        if len(incomplete) > 0:
            labels_incomplete.extend([fwd_map[i] for i in incomplete])
        tile.write(output, block, axes=axes_out)
        for k,v in polys.items():
            polys_all.setdefault(k,[]).append(v)
        label_offset += len(polys['prob'])

    polys_all = {k: (np.concatenate(v) if isinstance(v[0], np.ndarray) else v[0]) for k,v in polys_all.items()}

    if len(labels_incomplete) > 0:
        repaint_labels(output, labels_incomplete, polys_all)

    return output, polys_all, labels_incomplete



def _grid_divisible(grid, size, name=None, verbose=True):
    if size % grid == 0:
        return size
    _size = size
    size = math.ceil(size / grid) * grid
    if verbose:
        print(f"changing {'value' if name is None else name} from {_size} to {size} to be evenly divisible by {grid} (grid)")
    assert size % grid == 0
    return size



def render_polygons(polys, shape):
    # TODO: this function doesn't belong here
    # -> should really refactor polygons_to_label...
    assert isinstance(polys,dict) and all(k in polys for k in ('prob','coord','points'))
    from stardist import polygons_to_label
    ind = np.arange(len(polys['prob']),dtype=np.int)
    coord  = np.expand_dims(polys['coord'], 1)
    prob   = np.expand_dims(polys['prob'], 1)
    points = np.stack([ind,np.zeros_like(ind)],axis=-1)
    return polygons_to_label(coord, prob, points, shape=shape)



class Polygon:

    def __init__(self, coord, bbox=None, shape_max=None):
        self.bbox = self.coords_bbox(coord, shape_max=shape_max) if bbox is None else bbox
        self.coord = coord - np.array([r[0] for r in self.bbox]).reshape(2,1)
        self.slice = tuple(slice(*r) for r in self.bbox)
        self.shape = tuple(r[1]-r[0] for r in self.bbox)
        rr,cc = polygon(*self.coord, self.shape)
        self.mask = np.zeros(self.shape, np.bool)
        self.mask[rr,cc] = True

    @staticmethod
    def coords_bbox(*coords, shape_max=None):
        assert all(isinstance(c, np.ndarray) and c.ndim==2 and c.shape[0]==2 for c in coords)
        if shape_max is None:
            shape_max = (np.inf, np.inf)
        coord = np.concatenate(coords, axis=1)
        mins = np.maximum(0,         np.floor(np.min(coord,axis=1)) - 10).astype(int)
        maxs = np.minimum(shape_max, np.ceil (np.max(coord,axis=1)) + 10).astype(int)
        return tuple(zip(tuple(mins),tuple(maxs)))



class Polyhedron:

    def __init__(self, dist, origin, rays, bbox=None, shape_max=None):
        self.bbox = self.coords_bbox((dist, origin), rays=rays, shape_max=shape_max) if bbox is None else bbox
        self.slice = tuple(slice(*r) for r in self.bbox)
        self.shape = tuple(r[1]-r[0] for r in self.bbox)
        _origin = origin.reshape(1,3) - np.array([r[0] for r in self.bbox]).reshape(1,3)
        self.mask = polyhedron_to_label(dist[np.newaxis], _origin, rays, shape=self.shape, verbose=False).astype(np.bool)

    @staticmethod
    def coords_bbox(*dist_origin, rays, shape_max=None):
        dists, points = zip(*dist_origin)
        assert all(isinstance(d, np.ndarray) and d.ndim==1 and len(d)==len(rays) for d in dists)
        assert all(isinstance(p, np.ndarray) and p.ndim==1 and len(p)==3 for p in points)
        dists, points, verts = np.stack(dists)[...,np.newaxis], np.stack(points)[:,np.newaxis], rays.vertices[np.newaxis]
        coord = dists * verts + points
        coord = np.concatenate(coord, axis=0)
        if shape_max is None:
            shape_max = (np.inf, np.inf, np.inf)
        mins = np.maximum(0,         np.floor(np.min(coord,axis=0)) - 10).astype(int)
        maxs = np.minimum(shape_max, np.ceil (np.max(coord,axis=0)) + 10).astype(int)
        return tuple(zip(tuple(mins),tuple(maxs)))



def repaint_labels(output, labels, polys):
    assert output.ndim in (2,3)

    if output.ndim == 2:
        coord = lambda i: polys['coord'][i-1]
        prob  = lambda i: polys['prob'][i-1]

        for i in labels:
            poly_i = Polygon(coord(i), shape_max=output.shape)

            # find all labels that overlap with i (including i)
            overlapping = set(np.unique(output[poly_i.slice][poly_i.mask])) - {0}
            overlapping.add(i)
            # compute bbox union to find area to crop/replace in large output label image
            bbox_union = Polygon.coords_bbox(*[coord(j) for j in overlapping], shape_max=output.shape)

            # crop out label i, including the region that include all overlapping labels
            poly_i = Polygon(coord(i), bbox=bbox_union)
            mask = poly_i.mask.copy()

            # remove pixels from mask that belong to labels with higher probability
            for j in [j for j in overlapping if prob(j) > prob(i)]:
                mask[ Polygon(coord(j), bbox=bbox_union).mask ] = False

            crop = output[poly_i.slice]
            crop[crop==i] = 0 # delete all remnants of i in crop
            crop[mask]    = i # paint i where mask still active
    else:

        dist = lambda i: polys['dist'][i-1]
        origin = lambda i: polys['points'][i-1]
        prob = lambda i: polys['prob'][i-1]
        rays = polys['rays']

        for i in labels:
            poly_i = Polyhedron(dist(i), origin(i), rays, shape_max=output.shape)

            # find all labels that overlap with i (including i)
            overlapping = set(np.unique(output[poly_i.slice][poly_i.mask])) - {0}
            overlapping.add(i)
            # compute bbox union to find area to crop/replace in large output label image
            bbox_union = Polyhedron.coords_bbox(*[(dist(j),origin(j)) for j in overlapping], rays=rays, shape_max=output.shape)

            # crop out label i, including the region that include all overlapping labels
            poly_i = Polyhedron(dist(i), origin(i), rays, bbox=bbox_union)
            mask = poly_i.mask.copy()

            # remove pixels from mask that belong to labels with higher probability
            for j in [j for j in overlapping if prob(j) > prob(i)]:
                mask[ Polyhedron(dist(j), origin(j), rays, bbox=bbox_union).mask ] = False

            crop = output[poly_i.slice]
            crop[crop==i] = 0 # delete all remnants of i in crop
            crop[mask]    = i # paint i where mask still active
