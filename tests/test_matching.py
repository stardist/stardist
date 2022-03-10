import numpy as np
from stardist.matching import matching, _shuffle_labels, group_matching_labels
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


def test_grouping():

    y0 = test_image_nuclei_2d(return_mask=True)[1]
    y1 = _shuffle_labels(y0)
    assert np.allclose(y0>0,y1>0)
    assert all(np.allclose(sorted(u),sorted(v)) for u,v in
               zip(np.unique(y0,return_counts=True),np.unique(y1,return_counts=True)))
    assert matching(y0,y1, thresh=1).accuracy == 1

    ys = [y0,y1]
    ys_grouped = group_matching_labels(ys)
    assert np.allclose(ys_grouped[0],ys_grouped[1])

    ys = [y0]
    for i in range(5):
        ys.append(np.roll(_shuffle_labels(ys[-1]), 15))
    ys_grouped = group_matching_labels(ys)
    assert np.allclose(ys_grouped.max(axis=(1,2)), [183,199,215,231,247,263])

    return ys_grouped

        
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from stardist import random_label_cmap
    cmap = random_label_cmap()    

    y = test_grouping()
    fig, ax = plt.subplots(2,len(y)//2, figsize=(15,10))
    for a,_y in zip(ax.ravel(),y):
        a.imshow(_y, cmap=cmap, vmin=0, vmax=1000)
        a.imshow(_y==24, alpha=0.6, cmap='gray')
    plt.show()

    # y, y2 = test_matching()

