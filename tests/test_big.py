import sys
import numpy as np
import pytest

from csbdeep.utils import normalize
from stardist.matching import matching, relabel_sequential
from stardist import calculate_extents
from stardist.models import StarDist2D, StarDist3D
from utils import real_image2d, real_image3d, path_model2d, path_model3d

from stardist.big import get_tiling, predict_big, render_polygons



def repeat(mask, reps):
    if np.isscalar(reps):
        reps = (reps,) * mask.ndim
    def shift(mask, v):
        _mask = mask.copy()
        _mask[_mask>0] += v
        return _mask
    _shift = shift if np.issubdtype(mask.dtype, np.integer) else (lambda x, *args: x)
    for d,rep in enumerate(reps):
        n_labels = mask.max()
        mask = [_shift(mask, n_labels*i) for i in range(rep)]
        mask = np.concatenate(mask, axis=d)
    return mask



def reassemble(lbl, axes, tile_size, min_overlap, context, grid):
    tiles = get_tiling(lbl.shape, axes=axes, tile_size=tile_size, min_overlap=min_overlap, context=context, grid=grid)
    # print(len(tiles))
    result = np.zeros_like(lbl)

    for tile in tiles:
        block = tile.read(lbl)
        block = tile.crop_context(block)
        block = tile.filter_objects(block, polys=None)
        tile.write(result, block)

    assert np.all(lbl == result)



@pytest.mark.parametrize('grid', [1, 3, 6])
@pytest.mark.parametrize('tile_size, context', [(40,0), (55,3), (80,10), (128,17), (256,80), (512,93)])
def test_tiling2D(tile_size, context, grid):
    lbl = real_image2d()[1]
    lbl = lbl.astype(np.int32)

    max_sizes = tuple(calculate_extents(lbl, func=np.max))
    min_overlap = tuple(1+v for v in max_sizes)
    lbl = repeat(lbl, 4)
    assert max_sizes == tuple(calculate_extents(lbl, func=np.max))

    reassemble(lbl, 'YX', tile_size, min_overlap, context, grid)



@pytest.mark.parametrize('grid', [1, 3])
@pytest.mark.parametrize('tile_size, context', [((33,71,64),3), ((48,96,96),0), ((62,97,93),(0,11,9))])
def test_tiling3D(tile_size, context, grid):
    lbl = real_image3d()[1]
    lbl = lbl.astype(np.int32)

    max_sizes = tuple(calculate_extents(lbl, func=np.max))
    min_overlap = tuple(1+v for v in max_sizes)
    lbl = repeat(lbl, (2,4,4))
    assert max_sizes == tuple(calculate_extents(lbl, func=np.max))

    reassemble(lbl, 'ZYX', tile_size, min_overlap, context, grid)


@pytest.mark.parametrize('use_channel', [False, True])
def test_predict2D(use_channel):
    model_path = path_model2d()
    model = StarDist2D(None, name=model_path.name, basedir=str(model_path.parent))

    img = real_image2d()[0]
    img = normalize(img, 1, 99.8)
    img = repeat(img, 2)
    axes = 'YX'

    if use_channel:
        img = img[...,np.newaxis]
        axes += 'C'

    ref_labels, ref_polys = model.predict_instances(img)
    res_labels, res_polys, res_problems = predict_big(model, img, axes=axes, tile_size=288, min_overlap=32, context=96)
    assert len(res_problems) == 0

    m = matching(ref_labels, res_labels)
    assert (1.0, 1.0) == (m.accuracy, m.mean_true_score)

    m = matching(render_polygons(ref_polys, shape=img.shape),
                 render_polygons(res_polys, shape=img.shape))
    assert (1.0, 1.0) == (m.accuracy, m.mean_true_score)

    # sort them first lexicographic
    ref_inds = np.lexsort(ref_polys["points"].T)
    res_inds = np.lexsort(res_polys["points"].T)

    assert np.allclose(ref_polys["coord"][ref_inds],
                       res_polys["coord"][res_inds])
    assert np.allclose(ref_polys["points"][ref_inds],
                       res_polys["points"][res_inds])
    assert np.allclose(ref_polys["prob"][ref_inds],
                       res_polys["prob"][res_inds])

    return ref_polys, res_polys



def test_predict3D():
    model_path = path_model3d()
    model = StarDist3D(None, name=model_path.name, basedir=str(model_path.parent))

    img = real_image3d()[0]
    img = normalize(img, 1, 99.8)
    img = repeat(img, 2)

    ref_labels, ref_polys = model.predict_instances(img)
    res_labels, res_polys, res_problems = predict_big(model, img, axes='ZYX',
                                                      tile_size=(55,105,105), min_overlap=(13,25,25), context=(17,30,30))
    assert len(res_problems) == 0

    m = matching(ref_labels, res_labels)
    assert (1.0, 1.0) == (m.accuracy, m.mean_true_score)

    # compare 
    # sort them first lexicographic
    ref_inds = np.lexsort(ref_polys["points"].T)
    res_inds = np.lexsort(res_polys["points"].T)

    assert np.allclose(ref_polys["dist"][ref_inds],
                       res_polys["dist"][res_inds])
    assert np.allclose(ref_polys["points"][ref_inds],
                       res_polys["points"][res_inds])
    assert np.allclose(ref_polys["prob"][ref_inds],
                       res_polys["prob"][res_inds])

    return ref_polys, res_polys


if __name__ == '__main__':
    a, b = test_predict2D(False)
