from __future__ import absolute_import, print_function
from .version import __version__

# TODO: which functions to expose here? all?

from .nms import non_maximum_suppression, non_maximum_suppression_3d, non_maximum_suppression_3d_sparse
from .utils import edt_prob, fill_label_holes, sample_points, calculate_extents, export_imagej_rois, gputools_available
from .geometry import star_dist,   polygons_to_label,   relabel_image_stardist, ray_angles, dist_to_coord
from .geometry import star_dist3D, polyhedron_to_label, relabel_image_stardist3D
from .plot.plot import random_label_cmap, draw_polygons, _draw_polygons
from .plot.render import render_label, render_label_pred
from .rays3d import rays_from_json, Rays_Cartesian, Rays_SubDivide, Rays_Tetra, Rays_Octo, Rays_GoldenSpiral


def py35_deprecation():
    import sys
    from distutils.version import LooseVersion
    if sys.version_info[:2] == (3,5) and LooseVersion(__version__) < LooseVersion('0.6.0'):
        print("You are using Python 3.5, which will no longer be supported in the next major release of StarDist.\n"
              "→ Please upgrade to Python 3.6 or 3.7.", file=sys.stderr, flush=True)
py35_deprecation()
del py35_deprecation
