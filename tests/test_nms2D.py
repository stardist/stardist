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
def test_bbox_search(img):
    prob = edt_prob(img)
    dist = star_dist(img, n_rays=32, mode="cpp")
    points_a, prob_a, dist_a = non_maximum_suppression(
        dist, prob, prob_thresh=0.4, verbose=False, max_bbox_search=False)
    points_b, prob_b, dist_b = non_maximum_suppression(
        dist, prob, prob_thresh=0.4, verbose=False, max_bbox_search=True)
    check_similar(points_a, points_b)
    check_similar(prob_a, prob_b)
    check_similar(dist_a, dist_b)


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
    
    # points1 = _non_maximum_suppression_old(coord, prob, prob_thresh=0.9,
    #                                   grid = grid, 
    #                               nms_thresh=nms_thresh,
    #                               verbose=True)
    
    # points1[:,0] *= grid[0]
    # points1[:,1] *= grid[1]
    
    # points2, probi2, disti2 = non_maximum_suppression(dist, prob, grid = grid, prob_thresh=0.9,
    #                               nms_thresh=nms_thresh,
    #                               verbose=True)
    a = _non_maximum_suppression_old(coord, prob, prob_thresh=0.9,
                                      grid = grid, 
                                  nms_thresh=nms_thresh,
                                  verbose=True)
    
    
    b = non_maximum_suppression(dist, prob, grid = grid, prob_thresh=0.9,
                                  nms_thresh=nms_thresh,
                                  verbose=True)

    return a,b
    img1 = _polygons_to_label_old(coord, prob, points1, shape=shape)
    img2 = polygons_to_label(disti2, points2, shape=shape)


    np.allclose(points1, points2)
    np.allclose(img1, img2)
    return points1, img1, points2, img2


def test_pretrained():
    from stardist.models import StarDist2D
    img = normalize(real_image2d()[0])

    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    prob,dist = model.predict(img)
    y1, res1 = model._instances_from_prediction_old(img.shape,prob,dist, nms_thresh=.3) 
    y2, res2 = model._instances_from_prediction(img.shape,prob,dist, nms_thresh=.3)

    for k in res1.keys():
        assert np.allclose(res1[k],res2[k])
        
    assert np.allclose(y1,y2)
    
    return y1, res1, y2, res2


if __name__ == '__main__':

    # y1, res1, y2, res2 = test_pretrained()
    a,b = test_old_new((62,62),32,(2,2), nms_thresh = .39)
    

