import numpy as np
from stardist.affinity import dist_to_affinity2D
from stardist import star_dist
from utils import prob_dist_image2d
np.random.seed(42)
from scipy.ndimage import zoom
from stardist.geometry.geom2d import dist_to_coord, polygons_to_label
from stardist.nms import non_maximum_suppression
from skimage.morphology import watershed

if __name__ == '__main__':    


    img, prob, dist = prob_dist_image2d()

    grid = (2,2)
    
    coord = dist_to_coord(dist, grid = grid)
    points = non_maximum_suppression(coord, prob, grid = grid, prob_thresh=.4, nms_thresh=.3)

    labels0 = polygons_to_label(coord, prob, points, shape=img.shape)


    
    aff, aff_neg = dist_to_affinity2D(dist,
                                      weights = prob>0.03,
                                      grid = grid,
                                      normed=True, verbose = True);

    factor = tuple(s1/s2 for s1, s2 in zip(img.shape, prob.shape))
    
    potential = zoom(np.mean(aff,-1)-np.mean(aff_neg,-1), factor, order=1)

    markers = np.zeros(img.shape, np.int32)
    markers[grid[0]*points[:,0],grid[1]*points[:,1]] = np.arange(len(points))+1
    
    labels = watershed(-potential, markers=markers,mask = zoom(prob,factor, order=1)>0.01)
