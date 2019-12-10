import numpy as np
import pytest
from stardist import star_dist, edt_prob, non_maximum_suppression, dist_to_coord, polygons_to_label
from stardist.matching import matching
from utils import random_image, real_image2d, check_similar

def create_random_data(shape = (356,299), radius = 10, noise = .1, n_rays = 32):
    dist  = radius*np.ones(shape+(n_rays,))
    noise =  np.clip(noise, 0,1)
    if noise>0:
        dist *= (1+noise*np.random.uniform(-1,1, dist.shape))
    prob = np.random.uniform(0,1,shape)
    return prob, dist

def create_random_suppressed(shape = (356,299), radius = 10, noise = .1, n_rays = 32,nms_thresh = .4):
    prob, dist  = create_random_data(shape, radius, noise, n_rays)
    coord = dist_to_coord(dist)
    nms = non_maximum_suppression(coord, prob, prob_thresh=0.9,
                                  nms_thresh = nms_thresh,
                                  verbose=True)
    return nms

def test_large():
    nms = create_random_suppressed(shape=(2000,2007), radius=10, noise=0, nms_thresh = 0)
    return nms

@pytest.mark.parametrize('img', (real_image2d()[1], random_image((128, 123))))
def test_bbox_search(img):
    prob = edt_prob(img)
    dist = star_dist(img, n_rays=32, mode="cpp")
    coord = dist_to_coord(dist)
    nms_a = non_maximum_suppression(coord, prob, prob_thresh=0.4, verbose=False, max_bbox_search=False)
    nms_b = non_maximum_suppression(coord, prob, prob_thresh=0.4, verbose=False, max_bbox_search=True)
    check_similar(nms_a, nms_b)


@pytest.mark.parametrize('img', (real_image2d()[1], ))
def test_acc(img):
    prob = edt_prob(img)
    dist = star_dist(img, n_rays=32, mode="cpp")
    coord = dist_to_coord(dist)
    points = non_maximum_suppression(coord, prob, prob_thresh=0.4)
    img2 = polygons_to_label(coord, prob, points, shape=img.shape)
    m = matching(img, img2)
    acc = m.accuracy
    print("accuracy {acc:.2f}".format(acc=acc))
    assert acc > 0.9


if __name__ == '__main__':
    nms = test_large()
