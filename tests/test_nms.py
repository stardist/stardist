import numpy as np
from stardist import non_maximum_suppression

n=128

prob = np.random.uniform(0,1,(n,n))
coord = np.random.uniform(0,10,(n,n,2,32))

inds = non_maximum_suppression(coord, prob, nms_thresh =.3)
inds = non_maximum_suppression(coord, prob, nms_thresh =.5)
inds = non_maximum_suppression(coord, prob, nms_thresh =.7)
