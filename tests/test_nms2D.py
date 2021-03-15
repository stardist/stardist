import numpy as np
import pytest
from stardist import star_dist, edt_prob, non_maximum_suppression, dist_to_coord, polygons_to_label
from stardist.matching import matching
from csbdeep.utils import normalize
from utils import random_image, real_image2d, check_similar


def create_random_data(shape=(356, 299), radius=10, noise=.1, n_rays=32):
    dist = radius*np.ones(shape+(n_rays,))
    noise = np.clip(noise, 0, 1)
    if noise > 0:
        dist *= (1+noise*np.random.uniform(-1, 1, dist.shape))
    prob = np.random.uniform(0, 1, shape)
    return prob, dist


def create_random_suppressed(shape=(356, 299), grid = (1,1), radius=10, noise=.1, n_rays=32, nms_thresh=.1):
    prob, dist = create_random_data(shape, radius, noise, n_rays)
    prob = prob[::grid[0],::grid[1]]
    dist = dist[::grid[0],::grid[1]]
    points, probi, disti = non_maximum_suppression(dist, prob, prob_thresh=0.9,
                                  nms_thresh=nms_thresh,
                                  verbose=True)
    img = polygons_to_label(disti, points, prob=probi, shape=shape)
    return img

def test_large():
    nms = create_random_suppressed(
        shape=(2000, 2007), radius=10, noise=0, nms_thresh=0)
    return nms


@pytest.mark.parametrize('img', (real_image2d()[1], random_image((128, 123))))
def test_bbox_search_old(img):
    from stardist.geometry.geom2d import _polygons_to_label_old, _dist_to_coord_old
    from stardist.nms import _non_maximum_suppression_old

    prob = edt_prob(img)
    dist = star_dist(img, n_rays=32, mode="cpp")
    coord = _dist_to_coord_old(dist)
    points_a = _non_maximum_suppression_old(coord, prob, prob_thresh=0.4, verbose=False, max_bbox_search=False)
    points_b = _non_maximum_suppression_old(coord, prob, prob_thresh=0.4, verbose=False, max_bbox_search=True)
    img2_a = _polygons_to_label_old(coord, prob, points_a, shape=img.shape)
    img2_b = _polygons_to_label_old(coord, prob, points_b, shape=img.shape)
    check_similar(points_a, points_b)
    check_similar(img2_a, img2_b)


@pytest.mark.parametrize('grid', ((1,1),(2,2)))
@pytest.mark.parametrize('img', (real_image2d()[1], ))
def test_acc_old(img, grid):
    from stardist.geometry.geom2d import _polygons_to_label_old, _dist_to_coord_old
    from stardist.nms import _non_maximum_suppression_old

    prob = edt_prob(img)[::grid[0],::grid[1]]
    dist = star_dist(img, n_rays=32, mode="cpp")[::grid[0],::grid[1]]
    coord = _dist_to_coord_old(dist, grid = grid)
    points = _non_maximum_suppression_old(coord, prob, prob_thresh=0.4, grid=grid)
    img2 = _polygons_to_label_old(coord, prob, points, shape=img.shape)
    m = matching(img, img2)
    acc = m.accuracy
    print("accuracy {acc:.2f}".format(acc=acc))
    assert acc > 0.9

@pytest.mark.parametrize('grid', ((1,1),(2,2)))
@pytest.mark.parametrize('img', (real_image2d()[1], ))
def test_acc(img, grid):
    prob = edt_prob(img)[::grid[0],::grid[1]]
    dist = star_dist(img, n_rays=32, mode="cpp")[::grid[0],::grid[1]]
    points, probi, disti = non_maximum_suppression(dist, prob, grid = grid, prob_thresh=0.4)
    img2 = polygons_to_label(disti, points, shape=img.shape)
    m = matching(img, img2)
    acc = m.accuracy
    print("accuracy {acc:.2f}".format(acc=acc))
    assert acc > 0.9

@pytest.mark.parametrize('grid', ((1,1),(16,16)))
@pytest.mark.parametrize('n_rays', (11,32))
@pytest.mark.parametrize('shape', ((356, 299),(114, 217)))
def test_old_new(shape, n_rays, grid, radius=10, noise=.1, nms_thresh=.4):
    np.random.seed(42)
    from stardist.geometry.geom2d import _polygons_to_label_old, _dist_to_coord_old
    from stardist.nms import _non_maximum_suppression_old

    prob, dist = create_random_data(shape, n_rays = n_rays, radius=10, noise=.1)
    prob = prob[::grid[0],::grid[1]]
    dist = dist[::grid[0],::grid[1]]
    coord = _dist_to_coord_old(dist, grid = grid)
    inds1 = _non_maximum_suppression_old(coord, prob, prob_thresh=0.9,
                                         grid = grid,
                                         nms_thresh=nms_thresh,
                                         verbose=True)
    points1 = inds1*np.array(grid)
    sort_ind = np.argsort(prob[tuple(inds1.T)])[::-1]
    points1 = points1[sort_ind]

    points2, probi2, disti2 = non_maximum_suppression(dist, prob,
                                                      grid = grid, prob_thresh=0.9,
                                                      nms_thresh=nms_thresh,
                                                      verbose=True)

    img1 = _polygons_to_label_old(coord, prob, inds1, shape=shape)
    img2 = polygons_to_label(disti2, points2, shape=shape)


    assert len(points1) == len(points2)
    assert np.allclose(points1, points2)
    assert np.allclose(img1>0, img2>0)

    return points1, img1, points2, img2


def test_speed(nms_thresh = 0.3, grid = (1,1)):
    np.random.seed(42)
    from stardist.geometry.geom2d import _polygons_to_label_old, _dist_to_coord_old
    from stardist.nms import _non_maximum_suppression_old
    from time import time

    shape = (128,128)
    prob, dist = create_random_data(shape, n_rays = 32, radius=10, noise=.1)
    prob = np.tile(prob, (8,8))
    dist = np.tile(dist, (8,8,1))
    prob = prob[::grid[0],::grid[1]]
    dist = dist[::grid[0],::grid[1]]

    t1 = time()
    coord = _dist_to_coord_old(dist, grid = grid)
    points1 = _non_maximum_suppression_old(coord, prob, prob_thresh=0.9,
                                           grid = grid,
                                           nms_thresh=nms_thresh,
                                           verbose=True)
    t1 = time()-t1

    points1 = points1*np.array(grid)
    sort_ind = np.argsort(prob[tuple(points1.T)])[::-1]
    points1 = points1[sort_ind]

    t2 = time()

    points2, probi2, disti2 = non_maximum_suppression(dist, prob,
                                                      grid = grid, prob_thresh=0.9,
                                                      nms_thresh=nms_thresh,
                                                      use_kdtree = True,
                                                      verbose=True)
    t2 = time()-t2

    print("\n\n")
    print(f"old         : {t1:.2f}s")
    print(f"new (kdtree): {t2:.2f}s")

    return points1, points2


def bench():
    np.random.seed(42)
    from stardist.geometry.geom2d import _polygons_to_label_old, _dist_to_coord_old
    from stardist.nms import _non_maximum_suppression_old
    from time import time

    shape = (128,128)
    prob, dist = create_random_data(shape, n_rays = 32, radius=10, noise=.1)
    prob_thresh = 0.9

    def _f1(n):
        _prob = np.tile(prob, (n,n))
        _dist = np.tile(dist, (n,n,1))

        t = time()
        coord = _dist_to_coord_old(_dist)
        points1 = _non_maximum_suppression_old(coord, _prob, prob_thresh=prob_thresh,
                                               nms_thresh=.2,
                                               verbose=True)
        t = time()-t
        return np.count_nonzero(_prob>prob_thresh), t
    def _f2(n):
        _prob = np.tile(prob, (n,n))
        _dist = np.tile(dist, (n,n,1))

        t = time()
        points2, probi2, disti2 = non_maximum_suppression(_dist, _prob,
                                                          prob_thresh=prob_thresh,
                                                          nms_thresh=.2,
                                                          use_kdtree = True,
                                                      verbose=True)
        t = time()-t
        return np.count_nonzero(_prob>prob_thresh), t

    a1 = np.array(tuple(_f1(n) for n in range(1,20,2)))
    a2 = np.array(tuple(_f2(n) for n in range(1,20,2)))
    return a1, a2

if __name__ == '__main__':
    points1, img1, points2, img2 = test_old_new((62,82),32,(2,2), nms_thresh = .1)

