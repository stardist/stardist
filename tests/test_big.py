import sys
import numpy as np
import pytest

from stardist.matching import matching, relabel_sequential
from stardist import calculate_extents
from utils import real_image2d, real_image3d

from stardist.big import get_tiling



def repeat(mask, reps):
    if np.isscalar(reps):
        reps = (reps,) * mask.ndim
    def _shift(mask, v):
        _mask = mask.copy()
        _mask[_mask>0] += v
        return _mask
    for d,rep in enumerate(reps):
        n_labels = mask.max()
        mask = [_shift(mask, n_labels*i) for i in range(rep)]
        mask = np.concatenate(mask, axis=d)
    return mask



def reassemble(lbl, tile_size, min_overlap, context, grid):
    tiles = get_tiling(lbl.shape, tile_size=tile_size, min_overlap=min_overlap, context=context, grid=grid)
    # print(len(tiles))
    result = np.zeros_like(lbl)

    for tile in tiles:
        block = tile.read(lbl)
        block = tile.crop_context(block)
        block = tile.filter_objects(block)
        tile.write(result, block)

    assert np.all(lbl == result)



@pytest.mark.parametrize('grid', [1, 3, 8])
@pytest.mark.parametrize('tile_size, context', [(40,0), (55,3), (80,10), (128,17), (256,80), (512,93)])
def test_tiling2D(tile_size, context, grid):
    lbl = real_image2d()[1]
    lbl = lbl.astype(np.int32)

    max_sizes = tuple(calculate_extents(lbl, func=np.max))
    min_overlap = tuple(1+v for v in max_sizes)
    lbl = repeat(lbl, 4)
    assert max_sizes == tuple(calculate_extents(lbl, func=np.max))

    reassemble(lbl, tile_size, min_overlap, context, grid)



@pytest.mark.parametrize('grid', [1, 3, 8])
@pytest.mark.parametrize('tile_size, context', [((33,71,64),3), ((48,128,128),0), ((62,173,123),(9,27,34))])
def test_tiling3D(tile_size, context, grid):
    lbl = real_image3d()[1]
    lbl = lbl.astype(np.int32)

    max_sizes = tuple(calculate_extents(lbl, func=np.max))
    min_overlap = tuple(1+v for v in max_sizes)
    lbl = repeat(lbl, (2,4,4))
    assert max_sizes == tuple(calculate_extents(lbl, func=np.max))

    reassemble(lbl, tile_size, min_overlap, context, grid)


if __name__ == '__main__':
    pass
