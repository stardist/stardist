import numpy as np
import pytest
from stardist import non_maximum_suppression_3d, non_maximum_suppression_3d_sparse, polyhedron_to_label
from stardist import Rays_GoldenSpiral
from utils import random_image, check_similar
np.random.seed(42)

# def create_random_data(shape = (64,65,67), noise = .1, n_rays = 32, eps = (1,.6,.8)):
def create_random_data(shape = (64,65,67), noise = .1, n_rays = 32, eps = (1,1,1)):
    prob = np.random.uniform(0,1,shape)
    rays0 = Rays_GoldenSpiral(n_rays)
    dist = 10*np.broadcast_to(np.sum((rays0.vertices**2)*np.array(eps),1),shape+(n_rays,))
    rays = Rays_GoldenSpiral(n_rays, anisotropy = 1./np.array(eps))
    noise= np.clip(noise, 0,1)
    dist *= (1+noise*np.random.uniform(-1,1, dist.shape))
    return prob, dist, rays 

def create_random_suppressed(nms_thresh = .4, shape = (33,44,55), noise = .1, n_rays = 32):
    prob, dist, rays  = create_random_data(shape, noise, n_rays)
    
    points, probi, disti = non_maximum_suppression_3d(dist, prob, rays,
                                                      prob_thresh=0.9,
                                                      nms_thresh = nms_thresh,
                                                      verbose=True)
    return points, probi, disti, rays
    

@pytest.mark.parametrize('n_rays, nms_thresh, shape, noise',
                         [(5,  0, (33,44,55),  0),
                          (14,.2, (43,31,34), .1),
                          (22,.4, (33,44,55),  0),
                          (32,.6, (33,44,55), .1)])
def test_nms(n_rays, nms_thresh, shape,noise):
    points, probi, disti, rays = create_random_suppressed(nms_thresh, shape = shape, noise = 0, n_rays = n_rays)
    return points, rays, disti

def test_label():
    n_rays = 32 
    dist = 20*np.ones((1,n_rays))
    rays = Rays_GoldenSpiral(dist.shape[-1])
    points = [[20,20,20]]
    return polyhedron_to_label(dist,points, rays, shape = (33,44,55))


def test_nms_and_label(nms_thresh=0.1, shape = (33,44,55),noise = .1, n_rays = 32):
    points, probi, disti, rays = create_random_suppressed(nms_thresh, shape = shape, noise = noise, n_rays = n_rays)
    lbl = polyhedron_to_label(disti,points, rays, shape = shape)
    return lbl

@pytest.mark.parametrize('noise',(0,.2,.6,.9))
@pytest.mark.parametrize('n_rays',(32,65,100))
def test_nms_accuracy(noise, n_rays):
    dx = 3
    shape = (40,55,66)
    rays = Rays_GoldenSpiral(n_rays)
    dist = 10*(1+noise*np.sin(2*np.pi*rays.vertices[:,:2].T))
    points = [(20,20,20),(20+dx,20+2*dx,20-dx)]
    mask1 = polyhedron_to_label([dist[0]],[points[0]], rays, shape =shape)
    mask2 = polyhedron_to_label([dist[1]],[points[1]], rays, shape =shape)
    iou = np.count_nonzero(mask1*mask2)/min(np.count_nonzero(mask1), np.count_nonzero(mask2)+1e-10)
    prob= [1,.5]
    print("iou =", iou)
    sup1, _,_  = non_maximum_suppression_3d_sparse(dist,prob,points,
                                                   rays = rays,
                                                   nms_thresh = 0.95*iou,
                                                   verbose = True)
    sup2, _,_  = non_maximum_suppression_3d_sparse(dist,prob,points,
                                                   rays = rays,
                                                   nms_thresh = 1.15*iou,
                                                   verbose = True)
    print(len(sup1),len(sup2))
    assert len(sup1)==1 and len(sup2)==2
    return mask1, mask2


def test_speed(noises = (0,0.1,.2), n_rays = 32):
    from time import time

    def _bench(noise):
        t = time()
        create_random_suppressed(nms_thresh=.3, shape = (22,33,44), noise = noise, n_rays = n_rays)
        return time()-t

    ts = tuple(map(_bench, noises))
    for t, noise in zip(ts, noises):
        print("noise = {noise:.2f}\t t = {t:.2f} s".format(noise=noise,t=t))        
    return ts

@pytest.mark.parametrize('b', (None,2))
@pytest.mark.parametrize('grid', ((1,1,1),(1,2,4)))
def test_dense_sparse(b, grid):
    from stardist.nms import _ind_prob_thresh
    
    prob, dist, rays  = create_random_data(shape = (22,33,44), noise = .2, n_rays = 32)

    nms_thresh  = 0.3
    prob_thresh = 0.9
    
    points1, probi1, disti1 = non_maximum_suppression_3d(dist, prob, rays,
                                                         prob_thresh=prob_thresh,
                                                         grid = grid,
                                                         b = b, 
                                                         nms_thresh = nms_thresh,
                                                         verbose=True)

    inds   = _ind_prob_thresh(prob, prob_thresh, b=b)
    points = np.stack(np.where(inds), axis=1)
    points = (points * np.array(grid).reshape((1,3)))

    
    points2, probi2, disti2 = non_maximum_suppression_3d_sparse(dist[inds], prob[inds],
                                                    points, 
                                                    rays,
                                                    nms_thresh = nms_thresh,
                                                    verbose=True)
    assert np.allclose(points1, points2)
    assert np.allclose(probi1, probi2)
    assert np.allclose(disti1, disti2)
    return points1, points2

def test_special_case(noise=.4, dx = 9):
    n_rays = 66
    shape = (40,55,66)
    rays = Rays_GoldenSpiral(n_rays)
    # dist = 10*(1+noise*np.sin(2*np.pi*rays.vertices[:,:2].T))
    dist = np.broadcast_to(10*(1+noise*np.sin(2*np.pi*rays.vertices[:,2])),(2,n_rays))
    points = [(20,20,20),(20,20,20+dx)]
    prob = (.6,.4)
    non_maximum_suppression_3d_sparse(dist,prob,points,
                                      rays = rays,
                                      nms_thresh = .4,
                                      verbose = True)

    mask1 = polyhedron_to_label([dist[0]],[points[0]], rays, shape =shape)
    mask2 = polyhedron_to_label([dist[1]],[points[1]], rays, shape =shape)
    return mask1, mask2

if __name__ == '__main__':
    np.random.seed(42)
    # lbl = test_nms_and_label(.2,shape = (44,55,66), noise = .2)
    test_speed((0.4,))

    # p1, p2 = test_dense_sparse(b=None, grid = (1,1,1))

    # m1, m2 = test_nms_accuracy(noise=0.2, n_rays=66)
    # test_nms(20,.2,(50,50,50),0.6); 

    # lbl = test_nms_and_label(noise=0.1) 

    # m1, m2 = test_special_case(.4,9)
