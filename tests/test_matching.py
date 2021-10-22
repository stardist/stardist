import numpy as np
from stardist.matching import matching
from stardist.data import test_image_nuclei_2d
from skimage.draw import disk

def test_matching():
    
    width = 128
    n_circles = 5
    y = np.zeros((width,width), np.uint16)

    for i,r in enumerate(np.linspace(0, width, n_circles+2)[1:-1]):
        rr, cc = disk((width//2, r), radius=width//(3*n_circles), shape=y.shape)
        y[rr,cc] = i+1
    

    for shift in (0,5,10):
        
        y2 = np.roll(y, shift, axis=0 )

        iou = np.sum((y2==y)[y>0])/np.sum((y+y2)>0)
    
        res_all  = matching(y,y2, thresh=0.5*iou, report_matches=True)
        res_none = matching(y,y2, thresh=2.0*iou, report_matches=True)

        assert (res_all.tp,res_all.fp,res_all.fn)    == (n_circles,0,0)
        assert (res_none.tp,res_none.fp,res_none.fn) == (0,n_circles,n_circles)


        assert len(res_all.matched_pairs)==n_circles
        assert len(res_none.matched_pairs)==n_circles

    return y, y2

        
if __name__ == '__main__':


    y, y2 = test_matching()

