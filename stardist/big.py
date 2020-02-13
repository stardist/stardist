import numpy as np
import warnings
from tqdm import tqdm
from skimage.measure import regionprops
from csbdeep.utils import _raise
import math

from skimage.segmentation import clear_border
from collections import namedtuple
from itertools import product
from .matching import relabel_sequential



class Tile:

    def __init__(self, size, min_overlap, pred, context):
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
        self.succ = Tile(self.size, self.min_overlap, self, self.context)
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
    def slice(self):
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

        r_start = 0 if self.at_begin else (self.pred.overlap - self.context_start - self.pred.context_end)
        r_end = self.size - self.context_start - self.context_end

        # todo: of actual importance is whether one of those objects crosses the overlap region
        # assert bmax - bmin < self.min_overlap, 'found object bigger than min overlap'
        # if bmax - bmin < self.min_overlap:
        #     print('found object bigger than min overlap')
        assert not (not self.at_begin and bmin == 0 and bmax >= r_start), [(r_start,r_end), bbox, self]

        assert 0 <= bmin < bmax <= r_end
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

    def chain(self):
        tiles = [self]
        while not tiles[-1].at_end:
            tiles.append(tiles[-1].succ)
        return tiles

    def __iter__(self):
        return iter(self.chain())

    # ------------------------

    @staticmethod
    def get_tiling(size, tile_size, min_overlap, context, grid=1):

        assert 0 < grid <= min_overlap+2*context < tile_size <= size
        tile_size = _grid_divisible(grid, tile_size, name='tile_size')
        min_overlap = _grid_divisible(grid, min_overlap, name='min_overlap')
        context = _grid_divisible(grid, context, name='context')
        assert all(v % grid == 0 for v in (size, tile_size, min_overlap, context))

        # divide all sizes by grid
        size //= grid
        tile_size //= grid
        min_overlap //= grid
        context //= grid

        # compute tiling in grid-multiples
        t = first = Tile(tile_size, min_overlap, None, context)
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
            t = first = Tile(tile_size, min_overlap, None, context)
            t.stride = _t.stride*grid
            while not _t.at_end:
                _t = _t.succ
                t = t.add_succ()
                t.stride = _t.stride*grid
            last = t

        # for efficiency (to not determine starts recursively from now on)
        first.freeze()

        # sanity checks
        assert first.start == 0 and last.end == size
        assert all(t.overlap >= min_overlap for t in first)
        assert all (t.start % grid == 0 and t.end % grid == 0 for t in first)

        # print(); [print(t) for t in first]

        return first



class TileND:

    def __init__(self, id, tiles):
        self.id = id
        self.tiles = tuple(tiles)

    @property
    def slice(self):
        return tuple(t.slice for t in self.tiles)

    @property
    def slice_crop_context(self):
        return tuple(t.slice_crop_context for t in self.tiles)

    @property
    def slice_write(self):
        return tuple(t.slice_write for t in self.tiles)

    def is_responsible(self, slice):
        return all(t.is_responsible((s.start,s.stop)) for t,s in zip(self.tiles,slice))

    def __repr__(self):
        slices =  ','.join(f'{t.start:03}:{t.end:03}' for t in self.tiles)
        return f'{self.__class__.__name__}({self.id}|{slices})'

    def filter_objects(self, lbl):
        # todo: option to update lbl in-place
        assert np.issubdtype(lbl.dtype, np.integer)
        assert lbl.ndim == len(self.tiles)
        assert lbl.shape == tuple(s.stop-s.start for s in self.slice_crop_context)

        lbl_filtered = np.zeros_like(lbl)
        for r in regionprops(lbl, coordinates='rc'):
            slices = tuple(slice(r.bbox[i],r.bbox[i+lbl.ndim]) for i in range(lbl.ndim))
            if self.is_responsible(slices):
                lbl_filtered[slices][r.image] = r.label

        return lbl_filtered



def get_tiling(shape, tile_size, min_overlap, context, grid=1):
    shape = tuple(shape)
    n = len(shape)
    if np.isscalar(tile_size): tile_size = (tile_size,)*n
    if np.isscalar(min_overlap): min_overlap = (min_overlap,)*n
    if np.isscalar(context): context = (context,)*n
    if np.isscalar(grid): grid = (grid,)*n
    assert n == len(tile_size) == len(min_overlap) == len(context) == len(grid)

    tiling = [Tile.get_tiling(_size, _tile_size, _min_overlap, _context, _grid).chain()
              for _size, _tile_size, _min_overlap, _context, _grid
              in  zip(shape, tile_size, min_overlap, context, grid)]

    return tuple(TileND(i,tiles) for i,tiles in enumerate(product(*tiling)))



def predict_big(img, model, tile_size=16*16, min_overlap=4*16, context=5*16, grid=16):
    tiles = get_tiling(img.shape, tile_size, min_overlap, context, grid)
    output = np.zeros_like(img, dtype=np.int32)
    max_objects_per_block = 1000

    for tile in tiles:
        block, _ = model.predict_instances(img[tile.slice])
        block = block[tile.slice_crop_context]
        block = tile.filter_objects(block)
        block = relabel_sequential(block, 1 + tile.id*max_objects_per_block)[0]
        mask = block > 0
        output[tile.slice_write][mask] = block[mask]

    output = relabel_sequential(output)[0]
    return output



def _grid_divisible(grid, size, name=None):
    if size % grid == 0:
        return size
    _size = size
    size = math.ceil(size / grid) * grid
    print(f"changing {'value' if name is None else name} from {_size} to {size} to be evenly divisible by {grid} (grid)")
    assert size % grid == 0
    return size
