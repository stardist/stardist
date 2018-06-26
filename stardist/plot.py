from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np


def random_label_cmap(n=2**16):
    import matplotlib
    import colorsys
    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h,l,s = np.random.uniform(0,1,n), 0.4 + np.random.uniform(0,0.6,n), 0.2 + np.random.uniform(0,0.8,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)


def _plot_polygon(x,y,score,color):
    import matplotlib.pyplot as plt
    a,b = list(x),list(y)
    a += a[:1]
    b += b[:1]
    plt.plot(a,b,'--', alpha=1, linewidth=score, zorder=1, color=color)


def draw_polygons(coord, score, poly_idx, cmap=None, show_dist=False):
    """poly_idx is a N x 2 array with row-col coordinate indices"""
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = random_label_cmap(len(poly_idx)+1)

    assert len(cmap.colors[1:]) >= len(poly_idx)

    for p,c in zip(poly_idx,cmap.colors[1:]):
        plt.plot(p[1],p[0],'.r', alpha=.5+0.5, markersize=8*score[p[0],p[1]], color=c)

        if show_dist:
            for x,y in zip(coord[p[0],p[1],1], coord[p[0],p[1],0]):
                plt.plot((p[1],x),(p[0],y),'-',color=c, linewidth=0.4)

        _plot_polygon( coord[p[0],p[1],1], coord[p[0],p[1],0], 3*score[p[0],p[1]], color=c)
