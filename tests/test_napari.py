import sys
import numpy as np
from stardist.models import Config3D, StarDist3D
from stardist.data import  test_image_nuclei_2d
from utils import circle_image, real_image3d, path_model3d
from csbdeep.utils import normalize
from conftest import _model3d


def show_surface():

    import napari
    
    model = _model3d()
    img, mask = real_image3d()
    x = normalize(img, 1, 99.8)
    labels, polys = model.predict_instances(x)

    def surface_from_polys(polys): 
        from stardist.geometry import dist_to_coord3D 
        faces = polys["rays_faces"] 
        coord = dist_to_coord3D(polys["dist"], polys["points"], polys["rays_vertices"]) 
        faces = np.concatenate([faces+coord.shape[1]*i for i in np.arange(len(coord))]) 
        vertices = np.concatenate(coord, axis = 0) 
        values = np.concatenate([np.random.rand()*np.ones(len(c)) for c in coord]) 
        return (vertices,faces,values) 

    surface = surface_from_polys(polys)

    with napari.gui_qt(): 
        # add the surface
        viewer = napari.view_image(img) 
        viewer.add_surface(surface) 


def show_napari():
    import napari
    x = test_image_nuclei_2d()

    with napari.gui_qt():
        viewer =  napari.Viewer()

        # add the surface
        viewer.add_image(x)

        key = ("StarDist", "StarDist")  # (Plugin Name, Widget Name)
        viewer.window._add_plugin_dock_widget(key)

if __name__ == '__main__':

    show_napari()

