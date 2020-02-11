from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from csbdeep.utils import normalize
from ..utils import _normalize_grid
from ..matching import matching

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
    return _draw_polygons(polygons=coord[poly_idx[:,0],poly_idx[:,1]],
                         points=poly_idx,
                         scores=score[poly_idx[:,0],poly_idx[:,1]],
                         grid=grid, cmap=cmap, show_dist=show_dist)


def _draw_polygons(polygons, points=None, scores=None, grid=(1,1), cmap=None, show_dist=False):
    """
        polygons is a list/array of x,y coordinate lists/arrays
        points is a list/array of x,y coordinates
        scores is a list/array of scalar values between 0 and 1
    """
    # TODO: better name for this function?
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    grid = _normalize_grid(grid,2)
    if points is None:
        points = [None]*len(polygons)
    if scores is None:
        scores = np.ones(len(polygons))
    if cmap is None:
        cmap = random_label_cmap(len(polygons)+1)

    assert len(polygons) == len(scores)
    assert len(cmap.colors[1:]) >= len(polygons)
    assert not show_dist or all(p is not None for p in points)

    for point,poly,score,c in zip(points,polygons,scores,cmap.colors[1:]):
        if point is not None:
            plt.plot(point[1]*grid[1], point[0]*grid[0], '.', markersize=8*score, color=c)

        if show_dist:
            dist_lines = np.empty((poly.shape[-1],2,2))
            dist_lines[:,0,0] = poly[1]
            dist_lines[:,0,1] = poly[0]
            dist_lines[:,1,0] = point[1]*grid[1]
            dist_lines[:,1,1] = point[0]*grid[0]
            plt.gca().add_collection(LineCollection(dist_lines, colors=c, linewidths=0.4))

        _plot_polygon(poly[1], poly[0], 3*score, color=c)

