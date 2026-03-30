from __future__ import absolute_import, print_function

from .model2d import Config2D, StarDist2D, StarDistData2D
from .model3d import Config3D, StarDist3D, StarDistData3D

from csbdeep.utils import backend_channels_last
from csbdeep.utils.tf import BACKEND as K
if not backend_channels_last():
    raise NotImplementedError(
        "Keras is configured to use the '%s' image data format, which is currently not supported. "
        "Please change it to use 'channels_last' instead: "
        "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.image_data_format()
    )
del backend_channels_last, K

from csbdeep.models import register_model, register_aliases, clear_models_and_aliases
# register pre-trained models and aliases (TODO: replace with updatable solution)
clear_models_and_aliases(StarDist2D, StarDist3D)
register_model(StarDist2D,   '2D_versatile_fluo', 'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '4ad678d0758eed6e55625f1b5ae30771e59adb79f1239e09b9772eac8846c3dd')
register_model(StarDist2D,   '2D_versatile_he',   'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_he.zip', 'f1696ef0631bd7e1c0e5c0d3017e2b4c6a95e284c6aab9c22fc2f08317817b28')
register_model(StarDist2D,   '2D_paper_dsb2018',  'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_paper_dsb2018.zip', '4c11cf68512341d9e8ce3d1278c64ceb8ac400582739f85fcab079a2e82840d2')
register_model(StarDist2D,   '2D_demo',           'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_demo.zip', 'a1efaebd7103db6236655bf158b6e21cf5b38d58ec77a509802244a89a260fa4')
register_model(StarDist3D,   '3D_demo',           'https://github.com/stardist/stardist-models/releases/download/v0.1/python_3D_demo.zip', 'ea05831eb5acc8a2fd31eaa23f4460a196a9af53b14f40affb9d80885f699f90')

register_aliases(StarDist2D, '2D_paper_dsb2018',  'DSB 2018 (from StarDist 2D paper)')
register_aliases(StarDist2D, '2D_versatile_fluo', 'Versatile (fluorescent nuclei)')
register_aliases(StarDist2D, '2D_versatile_he',   'Versatile (H&E nuclei)')
del register_model, register_aliases, clear_models_and_aliases
