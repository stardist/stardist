import numpy as np
import pytest
from stardist import non_maximum_suppression_3d, polyhedron_to_label
from stardist import Rays_GoldenSpiral
from utils import random_image, check_similar

@pytest.mark.parametrize('n_rays, nms_thresh',[(5,0),(14,.2),(22,.4),(32,.6)])
def test_nms(n_rays, nms_thresh):
    dist = 10*np.ones((33,44,55,n_rays))
    prob = np.random.uniform(0,1,dist.shape[:3])
    rays = Rays_GoldenSpiral(dist.shape[-1])
    
    points, probi, disti = non_maximum_suppression_3d(dist, prob, rays,
                                                      prob_thresh=0.9,
                                                      nms_thresh = nms_thresh,
                                                      verbose=True)
    return points, rays, disti

def test_label():
    n_rays = 32 
    dist = 20*np.ones((1,n_rays))
    rays = Rays_GoldenSpiral(dist.shape[-1])
    points = [[20,20,20]]
    return polyhedron_to_label(dist,points, rays, shape = (33,44,55))



def test_nms_and_label(n_rays = 32, nms_thresh=0.1):
    points, rays, disti = test_nms(n_rays, nms_thresh)
    lbl = polyhedron_to_label(disti,points, rays, shape = (33,44,55))
    return lbl


if __name__ == '__main__':
    np.random.seed(42)
    lbl = test_nms_and_label()
