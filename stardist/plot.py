from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from .utils import _normalize_grid

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


def draw_polygons(coord, score, poly_idx, grid=(1,1), cmap=None, show_dist=False):
    """poly_idx is a N x 2 array with row-col coordinate indices"""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    grid = _normalize_grid(grid,2)
    if cmap is None:
        cmap = random_label_cmap(len(poly_idx)+1)

    assert len(cmap.colors[1:]) >= len(poly_idx)

    for p,c in zip(poly_idx,cmap.colors[1:]):
        plt.plot(p[1]*grid[1], p[0]*grid[0], '.', markersize=8*score[p[0],p[1]], color=c)

        if show_dist:
            # # too slow
            # for x,y in zip(coord[p[0],p[1],1], coord[p[0],p[1],0]):
            #     plt.plot((p[1]*grid[1],x), (p[0]*grid[0],y), '-', color=c, linewidth=0.4)
            dist_lines = np.empty((coord.shape[-1],2,2))
            dist_lines[:,0,0] = coord[p[0],p[1],1]
            dist_lines[:,0,1] = coord[p[0],p[1],0]
            dist_lines[:,1,0] = p[1]*grid[1]
            dist_lines[:,1,1] = p[0]*grid[0]
            plt.gca().add_collection(LineCollection(dist_lines, colors=c, linewidths=0.4))

        _plot_polygon(coord[p[0],p[1],1], coord[p[0],p[1],0], 3*score[p[0],p[1]], color=c)
