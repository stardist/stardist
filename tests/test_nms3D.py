import numpy as np
import pytest
from stardist import non_maximum_suppression_3d, polyhedron_to_label
from stardist import Rays_GoldenSpiral
from utils import random_image, check_similar

def test_nms():
    dist = 10*np.ones((33,44,55,32))
    prob = np.random.uniform(0,1,dist.shape[:3])
    rays = Rays_GoldenSpiral(dist.shape[-1])
    
    points, probi, disti = non_maximum_suppression_3d(dist, prob, rays,
                                                      prob_thresh=0.8,
                                                      nms_thresh = 0,
                                                      verbose=True)
    return points, rays, disti

def test_nms_and_label():
    points, rays, disti = test_nms()
    lbl = polyhedron_to_label(disti,points, rays, shape = (33,44,55))
    return lbl


if __name__ == '__main__':
    lbl = test_lbl()
