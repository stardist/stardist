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

def test_simple_nms():
    from stardist.lib.stardist2d import c_non_max_suppression_inds
    nx = 256
    # sample some radii
    n = 100
    nms_thresh = 0.4
    n_rays = 32
    rs = np.random.uniform(4,10, n)
    phi = np.linspace(0,2*np.pi,n_rays, endpoint = False)
    coords = np.stack([np.sin(phi), np.cos(phi)])
    coords = np.einsum("i,jk", rs, coords)
    coords += np.random.randint(nx//8,nx-nx//8,(n,2,1))
    scores = np.ones(n)
    mapping = np.empty((0,0),np.int32)

    survivors = c_non_max_suppression_inds(coords.astype(np.int32),
                    mapping, np.float32(nms_thresh), np.int32(0),
                    np.int32(1), np.int32(1),np.int32(0))
    

if __name__ == '__main__':
    pass
