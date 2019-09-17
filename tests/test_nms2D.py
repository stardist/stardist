import numpy as np
import pytest
from stardist import star_dist, edt_prob, non_maximum_suppression, dist_to_coord, polygons_to_label
from stardist.matching import matching
from utils import random_image, real_image2d, check_similar

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
    pass
